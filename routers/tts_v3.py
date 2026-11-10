"""
GPT-SoVITS API v3 - TTS 推理路由 (v3)

基于推理引擎的双队列架构：
- POST /api/v3/tts              提交推理任务，返回 task_id
- GET  /api/v3/tts/{id}         查询任务状态
- GET  /api/v3/tts/{id}/audio   获取完整音频（等待完成）
- WS   /api/v3/tts/stream       WebSocket 流式：提交+实时获取
- WS   /api/v3/tts/stream-input 双向流式：文本流入+音频流出
- GET  /api/v3/tts/queue        查看队列状态
"""
from __future__ import annotations

import asyncio
from typing import Optional, get_type_hints
from pathlib import Path

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import create_model

from api_neko.config import load_tts_voice, VoiceConfig, InferParams
from api_neko.audio_utils import join_wav_chunks


router = APIRouter(prefix="/api/v3", tags=["tts-v3"])


# ─── 请求模型 ───

TTSParameterOverrides = create_model(
    "TTSParameterOverrides",
    **{name: (Optional[annotation], None) for name, annotation in get_type_hints(InferParams).items()},
)


class TTSSubmitRequest(TTSParameterOverrides):
    """v3 TTS 提交请求"""
    text: str
    voice_id: str = "_default"
    media_type: Optional[str] = None


def _get_engine(request: Request):
    """从 app.state 获取推理引擎"""
    return request.app.state.inference_engine


def _build_tts_params(voice_cfg: VoiceConfig, text: str, overrides: dict) -> dict:
    """从声音配置 + 覆盖参数构建 TTS 推理参数"""
    params = TTSParameterOverrides.model_validate(overrides).model_dump(exclude_unset=True)
    return voice_cfg.to_tts_inputs(text, **params)


def _load_voice_config(voice_id: str) -> VoiceConfig:
    """加载声音配置"""
    return load_tts_voice(voice_id)


def _check_voice_backend(voice_cfg: VoiceConfig, request_or_app) -> str | None:
    """校验 voice 的 backend 字段与当前运行 backend 是否兼容。
    返回 None 表示兼容。不匹配时仅记录警告，不阻止请求
    （实际不兼容会在模型加载阶段自然报错）。"""
    voice_backend = voice_cfg.model.backend
    if not voice_backend:
        return None
    # Request 和 WebSocket 都有 .app 属性指向根 FastAPI app
    app = getattr(request_or_app, 'app', request_or_app)
    running_backend = getattr(app.state, 'backend_name', 'gsv')
    if voice_backend != running_backend:
        import logging
        logging.getLogger(__name__).warning(
            f"voice backend '{voice_backend}' 与 app.state.backend_name '{running_backend}' 不一致，"
            f"可能是 settings.toml 未正确配置或服务需要重启。"
        )
    return None


def _resolve_model_weights(voice_cfg: VoiceConfig) -> tuple[str, str]:
    """根据 voice 配置解析 gpt_weights 和 sovits_weights。
    Genie 模式下 onnx_model_dir 作为 sovits_weights 传递。"""
    gpt_w = voice_cfg.model.gpt_weights
    sovits_w = voice_cfg.model.onnx_model_dir or voice_cfg.model.sovits_weights
    return gpt_w, sovits_w


async def _wait_stream_end(websocket: WebSocket, sender_task: asyncio.Task) -> bool:
    """Watch disconnects while draining audio, including after the client's end."""
    async def disconnected():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((sender_task, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            return False
        await sender_task
        return True
    finally:
        for task in (watcher, sender_task):
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ─── REST 端点 ───

@router.post("/tts", summary="提交 TTS 推理任务")
async def tts_submit(request: Request, body: TTSSubmitRequest):
    """提交推理任务到队列，立即返回 task_id"""
    engine = _get_engine(request)

    # 加载声音配置
    try:
        voice_cfg = _load_voice_config(body.voice_id)
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"message": f"voice_id '{body.voice_id}' not found"})

    # 构建推理参数
    overrides = body.model_dump(exclude_unset=True)
    tts_params = _build_tts_params(voice_cfg, body.text, overrides)
    media_type = body.media_type or voice_cfg.output.media_type

    # 校验参数
    err = engine.validate_params({**tts_params, "media_type": media_type,
                                  "model_version": voice_cfg.model.version})
    if err:
        return JSONResponse(status_code=400, content={"message": err})

    # 校验 voice 后端兼容性
    backend_err = _check_voice_backend(voice_cfg, request)
    if backend_err:
        return JSONResponse(status_code=400, content={"message": backend_err})

    # 提交任务
    gpt_w, sovits_w = _resolve_model_weights(voice_cfg)
    task = await engine.submit(
        voice_id=body.voice_id,
        gpt_weights=gpt_w,
        sovits_weights=sovits_w,
        tts_params=tts_params,
        media_type=media_type,
    )

    return {
        "task_id": task.task_id,
        "timestamp": task.timestamp,
        "status": task.status.value,
        "queue_position": engine._queue.qsize(),
    }


@router.get("/tts/queue", summary="查看队列状态")
async def tts_queue_info(request: Request):
    """获取推理队列状态"""
    return _get_engine(request).get_queue_info()


@router.get("/tts/{task_id}", summary="查询任务状态")
async def tts_task_status(request: Request, task_id: int):
    """查询指定任务的状态"""
    task = _get_engine(request).get_task(task_id)
    if task is None:
        return JSONResponse(status_code=404, content={"message": f"task {task_id} not found"})

    return {
        "task_id": task.task_id,
        "timestamp": task.timestamp,
        "status": task.status.value,
        "voice_id": task.voice_id,
        "chunks_ready": len(task.output.chunks),
        "done": task.output.done,
        "error": task.output.error,
    }


@router.get("/tts/{task_id}/audio", summary="获取完整音频")
async def tts_task_audio(request: Request, task_id: int):
    """等待任务完成后返回完整音频。如果任务未完成，会阻塞等待（最多 120 秒）。"""
    task = _get_engine(request).get_task(task_id)
    if task is None:
        return JSONResponse(status_code=404, content={"message": f"task {task_id} not found"})

    # 等待任务完成
    timeout = 120.0
    elapsed = 0.0
    while not task.output.done and elapsed < timeout:
        await task.output.wait_for_update(timeout=5.0)
        elapsed += 5.0

    if task.output.error:
        return JSONResponse(status_code=500, content={"message": task.output.error})

    if not task.output.done:
        return JSONResponse(status_code=408, content={"message": "task timeout"})

    # 拼接所有 chunk
    chunks = [chunk.data for chunk in task.output.chunks]
    audio_data = join_wav_chunks(chunks) if task.media_type == "wav" else b"".join(chunks)
    return Response(audio_data, media_type=f"audio/{task.media_type}")


# ─── WebSocket 端点 ───

@router.websocket("/tts/stream")
async def tts_ws_stream(websocket: WebSocket):
    """
    WebSocket 流式推理。

    客户端发送 JSON:
    {
        "text": "...",
        "voice_id": "_default",
        "text_lang": "all_zh",
        ...（同 TTSSubmitRequest 字段）
    }

    服务端响应：
    1. JSON: {"type": "accepted", "task_id": 123, "timestamp": ...}
    2. JSON: {"type": "status", "status": "loading_model"} （如果需要切换模型）
    3. JSON: {"type": "status", "status": "inferring"}
    4. Binary: 音频 chunk 数据（逐个发送）
    5. JSON: {"type": "done", "chunks_total": N}
    或
    5. JSON: {"type": "error", "message": "..."}
    """
    await websocket.accept()
    task = None
    engine = websocket.app.state.inference_engine
    try:
        # 接收请求
        data = await websocket.receive_json()
        engine = websocket.app.state.inference_engine

        text = data.get("text", "")
        voice_id = data.get("voice_id", "_default")

        # 加载声音配置
        try:
            voice_cfg = _load_voice_config(voice_id)
        except FileNotFoundError:
            await websocket.send_json({"type": "error", "message": f"voice_id '{voice_id}' not found"})
            await websocket.close()
            return

        # 构建参数
        overrides = data
        tts_params = _build_tts_params(voice_cfg, text, overrides)
        media_type = data.get("media_type") or voice_cfg.output.media_type

        # 校验
        err = engine.validate_params({**tts_params, "media_type": media_type,
                                      "model_version": voice_cfg.model.version})
        if err:
            await websocket.send_json({"type": "error", "message": err})
            await websocket.close()
            return

        # 校验 voice 后端兼容性
        backend_err = _check_voice_backend(voice_cfg, websocket)
        if backend_err:
            await websocket.send_json({"type": "error", "message": backend_err})
            await websocket.close()
            return

        # 提交任务
        gpt_w, sovits_w = _resolve_model_weights(voice_cfg)
        task = await engine.submit(
            voice_id=voice_id,
            gpt_weights=gpt_w,
            sovits_weights=sovits_w,
            tts_params=tts_params,
            media_type=media_type,
            retain_result=False,
        )

        await websocket.send_json({
            "type": "accepted",
            "task_id": task.task_id,
            "timestamp": task.timestamp,
        })

        async def send_audio():
            sent_index = 0
            last_status = None
            while not task.output.done or sent_index < len(task.output.chunks):
                if task.status.value != last_status:
                    last_status = task.status.value
                    await websocket.send_json({"type": "status", "status": last_status})
                while sent_index < len(task.output.chunks):
                    await websocket.send_bytes(task.output.chunks[sent_index].data)
                    sent_index += 1
                if not task.output.done:
                    await task.output.wait_for_update(timeout=5.0)
            if task.output.error:
                await websocket.send_json({"type": "error", "message": task.output.error})
            else:
                await websocket.send_json({"type": "done", "chunks_total": sent_index})

        await _wait_stream_end(websocket, asyncio.create_task(send_audio()))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if task is not None:
            engine.cancel_task(task.task_id)
            engine.cleanup_task(task.task_id)
        try:
            await websocket.close()
        except Exception:
            pass


# ─── 双向流式 WebSocket：文本流入 + 音频流出 ───

# 句子边界标点（与 text_segmentation_method.py 中 splits 一致）
_SENTENCE_SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}


class _TextBuffer:
    """
    文本缓冲区：累积文本，检测句子边界，提取完整句子。

    调用 append(text) 追加文本，extract_sentences() 返回已完成的句子列表。
    flush() 返回缓冲区中所有剩余文本（即使不以标点结尾）。
    """

    def __init__(self):
        self._buf: str = ""

    def append(self, text: str):
        self._buf += text

    def extract_sentences(self) -> list[str]:
        """提取缓冲区中所有以标点结尾的完整句子"""
        sentences: list[str] = []
        # 从后往前找最后一个标点位置
        last_split = -1
        for i, ch in enumerate(self._buf):
            if ch in _SENTENCE_SPLITS:
                last_split = i

        if last_split < 0:
            return sentences

        # 截取到最后一个标点（含标点）
        completed = self._buf[: last_split + 1]
        self._buf = self._buf[last_split + 1 :]

        # 按标点切分为多个句子
        current = ""
        for ch in completed:
            current += ch
            if ch in _SENTENCE_SPLITS:
                s = current.strip()
                if s:
                    sentences.append(s)
                current = ""
        if current.strip():
            sentences.append(current.strip())

        return sentences

    def flush(self) -> str:
        """返回缓冲区中所有剩余文本并清空"""
        text = self._buf.strip()
        self._buf = ""
        return text

    @property
    def content(self) -> str:
        return self._buf


@router.websocket("/tts/stream-input")
async def tts_ws_stream_input(websocket: WebSocket):
    """Accept incremental text and return independent mono 48 kHz PCM WAV frames.

    The existing init/append/flush/end and ready/sentence/sentence_done/done
    messages are preserved. A sentence is announced only when its first audio
    is ready; an error before audio must not look like successful synthesis.
    """
    await websocket.accept()
    engine = websocket.app.state.inference_engine
    voice_cfg = None
    session_voice_id = "_default"
    overrides = {}
    text_buffer = _TextBuffer()
    task_queue = asyncio.Queue()
    owned_tasks = set()
    send_lock = asyncio.Lock()
    failed = False

    async def send_json(data):
        async with send_lock:
            await websocket.send_json(data)

    async def fail(message):
        nonlocal failed
        failed = True
        async with send_lock:
            await websocket.send_json({"type": "error", "message": str(message)})
            await websocket.close()

    async def submit_text(text):
        if not text.strip() or voice_cfg is None:
            return
        params = _build_tts_params(voice_cfg, text, overrides)
        gpt, sovits = _resolve_model_weights(voice_cfg)
        task = await engine.submit(
            voice_id=session_voice_id, gpt_weights=gpt, sovits_weights=sovits,
            tts_params=params, media_type="wav", pcm48_wav=True, retain_result=False,
        )
        owned_tasks.add(task.task_id)
        await task_queue.put(task)

    async def sender():
        current = None
        try:
            while True:
                current = await task_queue.get()
                if current is None:
                    return
                sent = 0
                while not current.output.done or sent < len(current.output.chunks):
                    if current.output.error:
                        await fail(current.output.error)
                        return
                    while sent < len(current.output.chunks):
                        if sent == 0:
                            await send_json({"type": "sentence", "text": current.tts_params["text"],
                                             "task_id": current.task_id})
                        async with send_lock:
                            await websocket.send_bytes(current.output.chunks[sent].data)
                        sent += 1
                    if not current.output.done:
                        await current.output.wait_for_update(timeout=5.0)
                if current.output.error:
                    await fail(current.output.error)
                    return
                await send_json({"type": "sentence_done", "task_id": current.task_id,
                                 "chunks_sent": sent})
                engine.cleanup_task(current.task_id)
                owned_tasks.discard(current.task_id)
                current = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not failed:
                try:
                    await fail(exc)
                except Exception:
                    pass

    sender_task = asyncio.create_task(sender())
    try:
        while not failed:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=300.0)
            cmd = data.get("cmd", "")
            if cmd == "init":
                if voice_cfg is not None:
                    raise ValueError("This connection is already initialized")
                session_voice_id = data.get("voice_id", "_default")
                voice_cfg = _load_voice_config(session_voice_id)
                error = _check_voice_backend(voice_cfg, websocket)
                if error:
                    raise ValueError(error)
                # Keep ready within the client's handshake timeout. This checks
                # configuration only; loading/inference stays on the shared worker.
                if not voice_cfg.ref_audio.path:
                    raise ValueError("ref_audio_path is required")
                paths = [voice_cfg.ref_audio.path, voice_cfg.model.gpt_weights,
                         voice_cfg.model.onnx_model_dir or voice_cfg.model.sovits_weights]
                for path in paths:
                    if path and not Path(voice_cfg._resolve_path(path)).exists():
                        raise ValueError(f"Configured file does not exist: {path}")
                if voice_cfg.model.version in ("v3", "v4", "v5dev", "v5turbo") and not voice_cfg.ref_audio.prompt_text.strip():
                    raise ValueError("prompt_text is required for V3/V4/V5")
                overrides = TTSParameterOverrides.model_validate(data).model_dump(exclude_unset=True)
                await send_json({"type": "ready", "voice_id": session_voice_id})
            elif cmd in ("text", "append", "flush"):
                if voice_cfg is None:
                    raise ValueError("not initialized, send init first")
                if cmd == "text":
                    await submit_text(data.get("data", ""))
                elif cmd == "append":
                    text_buffer.append(data.get("data", ""))
                    for sentence in text_buffer.extract_sentences():
                        await submit_text(sentence)
                else:
                    await submit_text(text_buffer.flush())
                    await send_json({"type": "flushed"})
            elif cmd == "end":
                if voice_cfg is not None:
                    await submit_text(text_buffer.flush())
                await task_queue.put(None)
                connected = await _wait_stream_end(websocket, sender_task)
                if connected and not failed:
                    await send_json({"type": "done"})
                break
            else:
                raise ValueError(f"unknown cmd: {cmd}")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if not failed:
            try:
                await fail(exc)
            except Exception:
                pass
    finally:
        # The initial validation socket has no owned jobs. It must not stop the
        # shared pipeline used by another client's actual speech connection.
        for task_id in owned_tasks:
            engine.cancel_task(task_id)
            engine.cleanup_task(task_id)
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
