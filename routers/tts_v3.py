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
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from api_neko.config import load_voice, load_default_config, VoiceConfig


router = APIRouter(prefix="/api/v3", tags=["tts-v3"])


# ─── 请求模型 ───

class TTSSubmitRequest(BaseModel):
    """v3 TTS 提交请求"""
    text: str
    voice_id: str = "_default"
    # 可选覆盖参数
    text_lang: Optional[str] = None
    speed_factor: Optional[float] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    batch_size: Optional[int] = None
    text_split_method: Optional[str] = None
    media_type: Optional[str] = None


def _get_engine(request: Request):
    """从 app.state 获取推理引擎"""
    return request.app.state.inference_engine


def _build_tts_params(voice_cfg: VoiceConfig, text: str, overrides: dict) -> dict:
    """从声音配置 + 覆盖参数构建 TTS 推理参数"""
    return voice_cfg.to_tts_inputs(text, **{k: v for k, v in overrides.items() if v is not None})


def _load_voice_config(voice_id: str) -> VoiceConfig:
    """加载声音配置"""
    if voice_id == "_default":
        return load_default_config()
    return load_voice(voice_id)


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
    overrides = {
        "text_lang": body.text_lang,
        "speed_factor": body.speed_factor,
        "temperature": body.temperature,
        "top_k": body.top_k,
        "top_p": body.top_p,
        "seed": body.seed,
        "batch_size": body.batch_size,
        "text_split_method": body.text_split_method,
    }
    tts_params = _build_tts_params(voice_cfg, body.text, overrides)
    media_type = body.media_type or voice_cfg.output.media_type

    # 校验参数
    err = engine.validate_params({**tts_params, "media_type": media_type})
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
    audio_data = b"".join(chunk.data for chunk in task.output.chunks)
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
        overrides = {k: data.get(k) for k in [
            "text_lang", "speed_factor", "temperature", "top_k", "top_p",
            "seed", "batch_size", "text_split_method",
        ]}
        tts_params = _build_tts_params(voice_cfg, text, overrides)
        media_type = data.get("media_type") or voice_cfg.output.media_type

        # 校验
        err = engine.validate_params({**tts_params, "media_type": media_type})
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
        )

        await websocket.send_json({
            "type": "accepted",
            "task_id": task.task_id,
            "timestamp": task.timestamp,
        })

        # 流式推送 chunk
        sent_index = 0
        last_status = None

        while not task.output.done or sent_index < len(task.output.chunks):
            # 推送状态变化
            if task.status.value != last_status:
                last_status = task.status.value
                await websocket.send_json({"type": "status", "status": last_status})

            # 推送新 chunk
            while sent_index < len(task.output.chunks):
                chunk = task.output.chunks[sent_index]
                await websocket.send_bytes(chunk.data)
                sent_index += 1

            # 如果还没完成，等待新数据
            if not task.output.done:
                await task.output.wait_for_update(timeout=5.0)

        # 发送完成或错误
        if task.output.error:
            await websocket.send_json({"type": "error", "message": task.output.error})
        else:
            await websocket.send_json({"type": "done", "chunks_total": len(task.output.chunks)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
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
    """
    双向流式 WebSocket：客户端高频发送文本片段，服务端自动切句并流式返回音频。
    接收和发送完全非阻塞，通过 task_queue 解耦。

    协议：
    ── 客户端 → 服务端（JSON）──
    1. {cmd:"init", voice_id:"...", text_lang:"...", ...overrides}  初始化配置
    2. {cmd:"text", data:"完整文本段落"}                             直接提交推理
    3. {cmd:"append", data:"碎片..."}                               追加到缓冲区（自动切句）
    4. {cmd:"flush"}                                                立即合成缓冲区剩余文本
    5. {cmd:"end"}                                                  结束，清空并关闭

    ── 服务端 → 客户端 ──
    1. {type:"ready"}                                               初始化成功
    2. {type:"sentence", text:"...", task_id:N}                     开始合成某句
    3. Binary: WAV chunk                                            音频数据
    4. {type:"sentence_done", task_id:N}                            某句合成完成
    5. {type:"flushed"}                                             flush 完成
    6. {type:"done"}                                                全部完成（end 后）
    7. {type:"error", message:"..."}                                错误
    """
    await websocket.accept()

    engine = websocket.app.state.inference_engine
    voice_cfg: Optional[VoiceConfig] = None
    session_voice_id: str = "_default"
    overrides: dict = {}
    text_buffer = _TextBuffer()

    # task_queue: 接收协程 → 发送协程 的通信通道
    # 放入 InferTask 表示有新任务，放入 None 表示会话结束
    task_queue: asyncio.Queue = asyncio.Queue()
    # send_lock: 防止并发写 WS（WebSocket 不支持并发 send）
    send_lock = asyncio.Lock()

    async def _safe_send_json(data: dict):
        async with send_lock:
            await websocket.send_json(data)

    async def _safe_send_bytes(data: bytes):
        async with send_lock:
            await websocket.send_bytes(data)

    async def _submit_text(text: str):
        """将一段文本提交为推理任务"""
        if not text.strip() or voice_cfg is None:
            return
        tts_params = _build_tts_params(voice_cfg, text, overrides)
        media_type = overrides.get("media_type") or voice_cfg.output.media_type

        gpt_w, sovits_w = _resolve_model_weights(voice_cfg)
        task = await engine.submit(
            voice_id=session_voice_id,
            gpt_weights=gpt_w,
            sovits_weights=sovits_w,
            tts_params=tts_params,
            media_type=media_type,
        )

        await _safe_send_json({
            "type": "sentence",
            "text": text,
            "task_id": task.task_id,
        })

        # 放入发送队列
        await task_queue.put(task)

    # ─── 发送协程：从 task_queue 取任务，按顺序推送音频 ───

    async def _sender():
        """后台协程：按顺序推送每个 task 的音频 chunk"""
        try:
            while True:
                task = await task_queue.get()
                if task is None:
                    break

                sent_index = 0
                while not task.output.done or sent_index < len(task.output.chunks):
                    while sent_index < len(task.output.chunks):
                        chunk = task.output.chunks[sent_index]
                        await _safe_send_bytes(chunk.data)
                        sent_index += 1

                    if not task.output.done:
                        await task.output.wait_for_update(timeout=5.0)

                if task.output.error:
                    await _safe_send_json({"type": "error", "message": task.output.error})
                else:
                    await _safe_send_json({
                        "type": "sentence_done",
                        "task_id": task.task_id,
                        "chunks_sent": sent_index,
                    })
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                await _safe_send_json({"type": "error", "message": f"sender error: {e}"})
            except Exception:
                pass

    # 启动发送协程
    sender_task = asyncio.create_task(_sender())

    # ─── 接收循环：处理客户端指令 ───

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=300.0)
            except asyncio.TimeoutError:
                await _safe_send_json({"type": "error", "message": "session timeout"})
                break

            cmd = data.get("cmd", "")

            if cmd == "init":
                voice_id = data.get("voice_id", "_default")
                try:
                    voice_cfg = _load_voice_config(voice_id)
                except FileNotFoundError:
                    await _safe_send_json({"type": "error", "message": f"voice_id '{voice_id}' not found"})
                    break

                # 校验 voice 后端兼容性
                backend_err = _check_voice_backend(voice_cfg, websocket)
                if backend_err:
                    await _safe_send_json({"type": "error", "message": backend_err})
                    break

                session_voice_id = voice_id
                overrides = {k: data.get(k) for k in [
                    "text_lang", "speed_factor", "temperature", "top_k", "top_p",
                    "seed", "batch_size", "text_split_method", "media_type",
                ] if data.get(k) is not None}

                await _safe_send_json({"type": "ready", "voice_id": voice_id})

            elif cmd == "text":
                # 直接提交整段文本进行推理（TTS 引擎内部会按 text_split_method 切分）
                if voice_cfg is None:
                    await _safe_send_json({"type": "error", "message": "not initialized, send init first"})
                    continue

                text_data = data.get("data", "").strip()
                if text_data:
                    await _submit_text(text_data)

            elif cmd == "append":
                # 碎片累积模式：追加到缓冲区，自动检测句子边界
                if voice_cfg is None:
                    await _safe_send_json({"type": "error", "message": "not initialized, send init first"})
                    continue

                text_buffer.append(data.get("data", ""))
                for sentence in text_buffer.extract_sentences():
                    await _submit_text(sentence)

            elif cmd == "flush":
                if voice_cfg is None:
                    continue
                remaining = text_buffer.flush()
                if remaining:
                    await _submit_text(remaining)
                # flushed 信号在所有 flush 产生的任务完成后由 sender 隐式完成
                # 这里放一个标记任务（None 不行，因为 None 是结束信号）
                # 改为：直接发 flushed，客户端通过 sentence_done 知道每句完成
                await _safe_send_json({"type": "flushed"})

            elif cmd == "end":
                print(f"[stream-input] end cmd received, queue size: {task_queue.qsize()}")
                if voice_cfg is not None:
                    remaining = text_buffer.flush()
                    if remaining:
                        await _submit_text(remaining)

                # 发送结束信号给 sender
                await task_queue.put(None)
                print("[stream-input] waiting for sender to finish...")
                # 等待 sender 完成所有推送
                await sender_task
                print("[stream-input] sender finished, sending done")
                await _safe_send_json({"type": "done"})
                break

            else:
                await _safe_send_json({"type": "error", "message": f"unknown cmd: {cmd}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await _safe_send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # 确保 sender 协程退出
        if not sender_task.done():
            await task_queue.put(None)
            sender_task.cancel()
            try:
                await sender_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await websocket.close()
        except Exception:
            pass
