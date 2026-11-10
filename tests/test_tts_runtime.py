import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from api_neko import config
from api_neko.backends.gsv_backend import GSVPipeline
from api_neko.inference import InferenceEngine, TaskStatus
from api_neko.routers import tts, tts_v3


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.loads = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.stop_calls = 0
        self.fail = False
        self.fragments = 1
        self.active = 0
        self.max_active = 0
        self.delay = 0
        self.model_paths = ("", "")
        self.models_used = []

    def init_t2s_weights(self, path):
        self.loads.append(("gpt", path))
        self.model_paths = path, self.model_paths[1]

    def init_vits_weights(self, path):
        self.loads.append(("sovits", path))
        self.model_paths = self.model_paths[0], path

    def validate_params(self, req):
        return None

    def get_model_paths(self):
        return self.model_paths

    def stop(self):
        self.stop_calls += 1

    def run(self, req):
        self.calls.append(dict(req))
        self.models_used.append(self.model_paths)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if not self.release.wait(3):
                raise RuntimeError("test gate timed out")
            if self.fail:
                raise ValueError("synthesis failed")
            for _ in range(self.fragments):
                if self.delay:
                    time.sleep(self.delay)
                yield 32000, np.full(3200, 1200, dtype=np.int16)
        finally:
            self.active -= 1


async def finished(task):
    async def wait():
        while not task.output.done:
            await task.output.wait_for_update(timeout=0.02)
    await asyncio.wait_for(wait(), timeout=4)


class EngineTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pipeline = FakePipeline()
        self.engine = InferenceEngine(self.pipeline, self.pipeline, [])
        await self.engine.start()

    async def asyncTearDown(self):
        self.pipeline.release.set()
        await self.engine.stop()

    async def submit(self, text="hello", **kwargs):
        return await self.engine.submit("example", "gpt", "sovits", {"text": text}, **kwargs)

    async def test_cancelled_work_finishes_physically_before_next_task(self):
        self.pipeline.release.clear()
        first = await self.submit("old", retain_result=False)
        self.assertTrue(await asyncio.to_thread(self.pipeline.started.wait, 2))
        queued = await self.submit("queued", retain_result=False)
        self.engine.cancel_task(queued.task_id)
        self.engine.cleanup_task(queued.task_id)
        self.engine.cancel_task(first.task_id)
        self.engine.cleanup_task(first.task_id)
        following = await self.submit("new")
        await asyncio.sleep(0.03)
        self.assertEqual(len(self.pipeline.calls), 1)
        self.assertFalse(first.output.done)
        self.pipeline.release.set()
        await finished(following)
        self.assertEqual([c["text"] for c in self.pipeline.calls], ["old", "new"])
        self.assertEqual(self.pipeline.stop_calls, 1)
        self.assertEqual(self.pipeline.max_active, 1)
        self.assertEqual(first.output.chunks, [])
        self.assertEqual(first.status, TaskStatus.CANCELLED)
        self.assertEqual(queued.status, TaskStatus.CANCELLED)
        self.assertNotIn(first.task_id, self.engine._tasks)
        self.assertNotIn(queued.task_id, self.engine._tasks)

    async def test_completed_rest_results_are_bounded(self):
        self.engine._max_completed_tasks = 2
        tasks = []
        for _ in range(5):
            task = await self.submit()
            tasks.append(task)
            await finished(task)
        self.assertEqual(list(self.engine._tasks), [tasks[-2].task_id, tasks[-1].task_id])

    async def test_shutdown_waits_for_active_thread(self):
        self.pipeline.release.clear()
        task = await self.submit()
        self.assertTrue(await asyncio.to_thread(self.pipeline.started.wait, 2))
        stopping = asyncio.create_task(self.engine.stop())
        await asyncio.sleep(0.03)
        self.assertFalse(stopping.done())
        self.pipeline.release.set()
        await asyncio.wait_for(stopping, 2)
        self.assertTrue(task.output.done)
        self.assertEqual(self.pipeline.active, 0)

    async def test_streaming_options_are_not_overwritten(self):
        task = await self.engine.submit("example", "", "", {
            "text": "hello", "streaming_mode": True, "return_fragment": True,
            "sample_steps": 4, "cfg_rate": 0, "use_cuda_graph": False,
        }, pcm48_wav=True)
        await finished(task)
        self.assertTrue(self.pipeline.calls[0]["streaming_mode"])
        self.assertEqual(self.pipeline.calls[0]["sample_steps"], 4)
        self.assertIs(self.pipeline.calls[0]["use_cuda_graph"], False)
        count = 0
        for chunk in task.output.chunks:
            with wave.open(BytesIO(chunk.data), "rb") as reader:
                self.assertEqual(reader.getframerate(), 48000)
                count += reader.getnframes()
        self.assertEqual(count, 4800)

    async def test_inference_failure_forces_fresh_model_resolution(self):
        self.pipeline.fail = True
        failed = await self.submit()
        await finished(failed)
        self.assertEqual(failed.status, TaskStatus.ERROR)
        self.pipeline.fail = False
        recovered = await self.submit()
        await finished(recovered)
        self.assertEqual(recovered.status, TaskStatus.DONE)
        self.assertEqual(self.pipeline.loads.count(("sovits", "sovits")), 2)
        self.assertEqual(self.pipeline.loads.count(("gpt", "gpt")), 2)

    async def test_manual_model_selection_and_v3_keep_their_own_weights(self):
        first = await self.engine.submit("A", "gpt-A", "A", {"text": "first"})
        await finished(first)
        await self.engine.select_model_weights(gpt="gpt-B", sovits="B")
        second = await self.engine.submit("A", "gpt-A", "A", {"text": "second"})
        await finished(second)
        manual = await self.engine.submit_selected({"text": "manual"})
        await finished(manual)
        self.assertEqual(self.pipeline.models_used, [("gpt-A", "A"), ("gpt-A", "A"), ("gpt-B", "B")])

    async def test_v2_inference_cannot_overlap_the_v3_worker(self):
        self.pipeline.release.clear()
        first = await self.submit("v3")
        self.assertTrue(await asyncio.to_thread(self.pipeline.started.wait, 2))
        legacy = asyncio.create_task(tts.tts_handle({"text": "v2", "streaming_mode": False, "media_type": "wav"}, self.engine))
        await asyncio.sleep(0.03)
        self.assertFalse(legacy.done())
        self.assertEqual(len(self.pipeline.calls), 1)
        self.pipeline.release.set()
        self.assertEqual((await legacy).status_code, 200)
        await finished(first)
        self.assertEqual(self.pipeline.max_active, 1)

    async def test_output_deadline_does_not_count_update_notifications_as_seconds(self):
        self.pipeline.fragments = 35
        self.pipeline.delay = 0.01
        task = await self.submit()
        self.assertFalse(await task.output.wait_until_done(timeout=0.02))
        self.assertTrue(await task.output.wait_until_done(timeout=2))
        self.assertEqual(len(task.output.chunks), 35)


class ModelSwitchTests(TestCase):
    def test_validation_uses_requested_version_before_weights_switch(self):
        pipeline = GSVPipeline.__new__(GSVPipeline)
        pipeline._tts_config = SimpleNamespace(version="v5turbo", v1_languages=["en"], v2_languages=["all_zh"])
        pipeline._cut_method_names = ["cut5"]
        request = {"text": "hello", "ref_audio_path": "example.wav", "text_lang": "all_zh",
                   "prompt_lang": "all_zh", "model_version": "v2ProPlus"}
        self.assertIsNone(pipeline.validate_params(request))
        request["model_version"] = "v5turbo"
        self.assertIn("prompt_text", pipeline.validate_params(request))

    def test_changing_weights_invalidates_reference_features(self):
        pipeline = GSVPipeline.__new__(GSVPipeline)
        pipeline._tts = SimpleNamespace(init_vits_weights=Mock(), stop=Mock(),
            prompt_cache={"ref_audio_path": "same.wav", "prompt_text": "same text"})
        pipeline.init_vits_weights("new.pth")
        self.assertIsNone(pipeline._tts.prompt_cache["ref_audio_path"])
        self.assertIsNone(pipeline._tts.prompt_cache["prompt_text"])
        pipeline.stop()
        pipeline._tts.stop.assert_called_once()


class ProtocolTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference.wav"
        self.reference.write_bytes(b"fake reference, inference is mocked")
        self.voice = config.VoiceConfig()
        self.voice.voice.id = "example-v5turbo"
        self.voice.model.version = "v5turbo"
        self.voice.ref_audio.path = str(self.reference)
        self.voice.ref_audio.prompt_text = "reference text"
        self.pipeline = FakePipeline()
        self.engine = InferenceEngine(self.pipeline, self.pipeline, [])

        @asynccontextmanager
        async def lifespan(app):
            await self.engine.start()
            try:
                yield
            finally:
                self.pipeline.release.set()
                await self.engine.stop()

        app = FastAPI(lifespan=lifespan)
        app.state.inference_engine = self.engine
        app.state.tts_pipeline = self.pipeline
        app.state.backend_name = "gsv"
        app.include_router(tts_v3.router)
        app.include_router(tts.router)
        self.patcher = patch.object(tts_v3, "_load_voice_config", return_value=self.voice)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def open_stream(self, **overrides):
        socket = self.client.websocket_connect("/api/v3/tts/stream-input")
        socket.__enter__()
        self.addCleanup(socket.__exit__, None, None, None)
        socket.send_json({"cmd": "init", "voice_id": self.voice.voice.id, **overrides})
        self.assertEqual(socket.receive_json()["type"], "ready")
        return socket

    def receive_completed(self, socket):
        events, audio = [], []
        while True:
            message = socket.receive()
            if message.get("bytes") is not None:
                audio.append(message["bytes"])
                events.append("audio")
            else:
                event = json.loads(message["text"])
                events.append(event["type"])
                if event["type"] == "done":
                    break
        self.assertEqual(socket.receive()["type"], "websocket.close")
        return events, audio

    def test_unchanged_neko_messages_receive_complete_pcm48_wav(self):
        socket = self.open_stream(sample_steps=4, cfg_rate=0.4, use_flash_attention=False)
        socket.send_json({"cmd": "append", "data": "hello"})
        socket.send_json({"cmd": "append", "data": "."})
        socket.send_json({"cmd": "end"})
        events, chunks = self.receive_completed(socket)
        self.assertEqual(events[0], "sentence")
        self.assertEqual(events[-2:], ["sentence_done", "done"])
        frames = 0
        for chunk in chunks:
            with wave.open(BytesIO(chunk), "rb") as reader:
                self.assertEqual((reader.getframerate(), reader.getnchannels(), reader.getsampwidth()), (48000, 1, 2))
                frames += reader.getnframes()
        self.assertEqual(frames, 4800)
        self.assertEqual(self.pipeline.calls[0]["sample_steps"], 4)
        self.assertEqual(self.pipeline.calls[0]["cfg_rate"], 0.4)
        self.assertIs(self.pipeline.calls[0]["use_flash_attention"], False)
        self.assertEqual(len(self.engine._tasks), 0)

    def test_failure_before_audio_is_not_announced_as_a_sentence(self):
        self.pipeline.fail = True
        socket = self.open_stream()
        socket.send_json({"cmd": "append", "data": "hello."})
        socket.send_json({"cmd": "end"})
        result = socket.receive_json()
        self.assertEqual(result["type"], "error")
        self.assertIn("synthesis failed", result["message"])
        self.assertEqual(socket.receive()["type"], "websocket.close")

    def test_disconnect_cancels_owned_work_but_idle_probe_does_not(self):
        self.pipeline.release.clear()
        with self.client.websocket_connect("/api/v3/tts/stream-input") as socket:
            socket.send_json({"cmd": "init", "voice_id": self.voice.voice.id})
            self.assertEqual(socket.receive_json()["type"], "ready")
            socket.send_json({"cmd": "append", "data": "old."})
            self.assertTrue(self.pipeline.started.wait(2))
        # Wait for the server-side disconnect cleanup to run on the app loop.
        self.client.portal.call(asyncio.sleep, 0.03)
        self.assertEqual(self.pipeline.stop_calls, 1)
        with self.client.websocket_connect("/api/v3/tts/stream-input") as probe:
            probe.send_json({"cmd": "init", "voice_id": self.voice.voice.id})
            self.assertEqual(probe.receive_json()["type"], "ready")
        self.client.portal.call(asyncio.sleep, 0.03)
        self.assertEqual(self.pipeline.stop_calls, 1)
        self.pipeline.release.set()

    def test_disconnect_after_end_is_observed_before_audio_is_ready(self):
        self.pipeline.release.clear()
        socket = self.open_stream()
        socket.send_json({"cmd": "append", "data": "old."})
        socket.send_json({"cmd": "end"})
        self.assertTrue(self.pipeline.started.wait(2))
        socket.send({"type": "websocket.disconnect", "code": 1000})
        self.client.portal.call(asyncio.sleep, 0.03)
        self.assertEqual(self.pipeline.stop_calls, 1)
        self.assertEqual(self.pipeline.active, 1)
        self.pipeline.release.set()

    def test_single_request_stream_observes_disconnect_during_inference(self):
        self.pipeline.release.clear()
        with self.client.websocket_connect("/api/v3/tts/stream") as socket:
            socket.send_json({"text": "old", "voice_id": self.voice.voice.id, "sample_steps": 4})
            self.assertEqual(socket.receive_json()["type"], "accepted")
            self.assertTrue(self.pipeline.started.wait(2))
            socket.send({"type": "websocket.disconnect", "code": 1000})
            self.client.portal.call(asyncio.sleep, 0.03)
            self.assertEqual(self.pipeline.stop_calls, 1)
        self.pipeline.release.set()

    def test_rest_keeps_source_format_and_joins_fragment_headers(self):
        self.pipeline.fragments = 2
        response = self.client.post("/api/v3/tts", json={"text": "hello", "voice_id": self.voice.voice.id,
            "sample_steps": 4, "use_cuda_graph": False, "return_fragment": True})
        self.assertEqual(response.status_code, 200)
        task_id = response.json()["task_id"]
        result = self.client.get(f"/api/v3/tts/{task_id}/audio")
        self.assertEqual(result.status_code, 200)
        with wave.open(BytesIO(result.content), "rb") as reader:
            self.assertEqual(reader.getframerate(), 32000)
            self.assertEqual(reader.getnframes(), 6400)
        self.assertIs(self.pipeline.calls[0]["use_cuda_graph"], False)
        self.assertTrue(self.pipeline.calls[0]["return_fragment"])

    def test_v2_get_and_post_preserve_model_defaults(self):
        fields = {"text": "hello", "text_lang": "all_zh", "ref_audio_path": str(self.reference),
                  "prompt_text": "reference", "prompt_lang": "all_zh", "use_cuda_graph": False}
        for method in ("get", "post"):
            response = getattr(self.client, method)("/api/v2/tts", **{"params" if method == "get" else "json": fields})
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(self.pipeline.calls[-1]["sample_steps"])
            self.assertIs(self.pipeline.calls[-1]["use_cuda_graph"], False)

    def test_many_rest_fragments_do_not_trigger_a_premature_120_second_timeout(self):
        self.pipeline.fragments = 35
        self.pipeline.delay = 0.01
        response = self.client.post("/api/v3/tts", json={"text": "many fragments", "voice_id": self.voice.voice.id})
        result = self.client.get(f"/api/v3/tts/{response.json()['task_id']}/audio")
        self.assertEqual(result.status_code, 200)
        with wave.open(BytesIO(result.content), "rb") as reader:
            self.assertEqual(reader.getnframes(), 35 * 3200)

    def test_manual_weight_endpoint_keeps_v3_model_identity_consistent(self):
        selected = self.client.get("/api/v2/set_sovits_weights", params={"weights_path": "model-B"})
        self.assertEqual(selected.status_code, 200)
        self.voice.model.sovits_weights = "model-A"
        response = self.client.post("/api/v3/tts", json={"text": "model A", "voice_id": self.voice.voice.id})
        self.assertEqual(self.client.get(f"/api/v3/tts/{response.json()['task_id']}/audio").status_code, 200)
        self.assertEqual(self.pipeline.models_used[-1][1], "model-A")
        response = self.client.post("/api/v2/tts", json={"text": "manual B"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.pipeline.models_used[-1][1], "model-B")

    def test_v2_stream_keeps_one_wav_header_and_cleans_its_task(self):
        self.pipeline.fragments = 2
        result = self.client.post("/api/v2/tts", json={"text": "stream", "streaming_mode": 1, "media_type": "wav"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content[:4], b"RIFF")
        self.assertEqual(result.content.count(b"RIFF"), 1)
        self.assertEqual(len(result.content), 44 + 2 * 3200 * 2)
        self.assertEqual(len(self.engine._tasks), 0)

    def test_v2_stream_reports_failure_before_sending_audio_headers(self):
        self.pipeline.fail = True
        result = self.client.post("/api/v2/tts", json={"text": "fail", "streaming_mode": 1})
        self.assertEqual(result.status_code, 400)
        self.assertIn("synthesis failed", result.json()["Exception"])
        self.assertEqual(len(self.engine._tasks), 0)
