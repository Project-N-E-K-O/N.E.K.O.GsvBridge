"""Optional real-host test. No local voices, settings or weights are written."""
from contextlib import asynccontextmanager
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import TestCase, skipUnless
from unittest.mock import patch
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from api_neko import config
from api_neko.backends.gsv_backend import GSVPipeline
from api_neko.inference import InferenceEngine
from api_neko.routers import tts, tts_v3


@skipUnless(os.environ.get("GSV_TEST_HOST_ROOT"), "set GSV_TEST_HOST_ROOT for real GPU inference")
class HostIntegrationTests(TestCase):
    def test_model_switches_keep_the_existing_neko_protocol(self):
        import yaml
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable in this test process; GPU regression was not run")
        host = Path(os.environ["GSV_TEST_HOST_ROOT"]).resolve()
        repository = Path(__file__).resolve().parents[1]
        old_cwd = Path.cwd()
        old_path = list(sys.path)
        # The tracked public sample is the only reference used by this test.
        public = config._load_raw_toml(repository / "voices/init.toml")
        reference = repository / "voices" / Path(public["ref_audio"]["path"]).name
        profiles = {
            "v2ProPlus": "v2Pro/s2Gv2ProPlus.pth",
            "v5turbo": "gsv-v5-pretrained/s2Gv5turbo.pth",
            "v5dev": "gsv-v5-pretrained/s2Gv5dev.pth",
        }
        gpt = host / "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        models = {key: host / "GPT_SoVITS/pretrained_models" / value for key, value in profiles.items()}
        for path in [gpt, reference, *models.values()]:
            self.assertTrue(path.is_file(), str(path))
        try:
            sys.path[:0] = [str(host), str(host / "GPT_SoVITS")]
            os.chdir(host)
            with tempfile.TemporaryDirectory() as directory:
                infer_config = Path(directory) / "tts_infer.yaml"
                infer_config.write_text(yaml.safe_dump({"custom": {
                    "device": "cuda", "is_half": True, "version": "v2ProPlus",
                    "t2s_weights_path": str(gpt), "vits_weights_path": str(models["v2ProPlus"]),
                    "bert_base_path": str(host / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
                    "cnhuhbert_base_path": str(host / "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
                }}), encoding="utf-8")
                pipeline = GSVPipeline(str(infer_config))
                engine = InferenceEngine(pipeline, pipeline, [])

                @asynccontextmanager
                async def lifespan(app):
                    await engine.start()
                    try:
                        yield
                    finally:
                        await engine.stop()

                app = FastAPI(lifespan=lifespan)
                app.state.inference_engine = engine
                app.state.backend_name = "gsv"
                app.include_router(tts_v3.router)
                app.include_router(tts.router)

                def voice_for(version):
                    data = deepcopy(public)
                    data["voice"]["id"] = version
                    data["model"].update(version=version, gpt_weights=str(gpt), sovits_weights=str(models[version]))
                    data["ref_audio"]["path"] = str(reference)
                    data["params"].pop("sample_steps", None)
                    data["params"].update(seed=12345, batch_size=1)
                    return config._dict_to_voice_config(data)

                with patch.object(tts_v3, "_load_voice_config", side_effect=voice_for), TestClient(app) as client:
                    for version in ("v2ProPlus", "v5turbo", "v5dev", "v2ProPlus"):
                        samples = []
                        with client.websocket_connect("/api/v3/tts/stream-input") as socket:
                            socket.send_json({"cmd": "init", "voice_id": version})
                            self.assertEqual(socket.receive_json()["type"], "ready")
                            for text in ("你好，", "这是模型兼容性测试。"):
                                socket.send_json({"cmd": "append", "data": text})
                            socket.send_json({"cmd": "end"})
                            while True:
                                message = socket.receive()
                                if message.get("bytes") is not None:
                                    self.assertLessEqual(len(message["bytes"]), 96044)
                                    with wave.open(BytesIO(message["bytes"]), "rb") as reader:
                                        self.assertEqual((reader.getframerate(), reader.getnchannels(), reader.getsampwidth()), (48000, 1, 2))
                                        samples.append(np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16))
                                else:
                                    event = json.loads(message["text"])
                                    self.assertNotEqual(event["type"], "error", event)
                                    if event["type"] == "done":
                                        break
                            self.assertEqual(socket.receive()["type"], "websocket.close")
                        audio = np.concatenate(samples)
                        self.assertTrue(np.any(audio))
                        self.assertEqual(len(engine._tasks), 0)
                        print(json.dumps({"profile": version, "pcm_samples": int(audio.size), "sample_rate": 48000}))
                        if version == "v5dev":
                            # Simulate the model-management page selecting Turbo.
                            # The next v3 request still asks for V2ProPlus.
                            selected = client.get("/api/v2/set_sovits_weights", params={"weights_path": str(models["v5turbo"])})
                            self.assertEqual(selected.status_code, 200)
                    legacy = client.post("/api/v2/tts", json={
                        "text": "模型兼容性测试。", "text_lang": "all_zh", "ref_audio_path": str(reference),
                        "prompt_text": public["ref_audio"]["prompt_text"],
                        "prompt_lang": public["ref_audio"]["prompt_lang"], "seed": 12345,
                    })
                    self.assertEqual(legacy.status_code, 200)
                    self.assertEqual(pipeline.get_info().version, "v5turbo")
                    with wave.open(BytesIO(legacy.content), "rb") as reader:
                        self.assertEqual(reader.getframerate(), 48000)
                        self.assertGreater(reader.getnframes(), 0)
                    self.assertEqual(len(engine._tasks), 0)
                    print(json.dumps({"profile": "v2-manual-turbo-after-v3", "sample_rate": 48000, "restored_selection": True}))
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path
