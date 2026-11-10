"""
GPT-SoVITS API v3 - TTS 推理路由

从 api_v2.py 移植，提供 TTS 推理和模型切换端点。
路由前缀: /api/v2（保持与 v2 API 兼容的参数格式）
"""
from __future__ import annotations

from typing import Union

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from api_neko.audio_utils import join_wav_chunks, wave_header_chunk


router = APIRouter(prefix="/api/v2", tags=["tts"])


# ─── 请求模型 ───

class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: Union[int, None] = None
    cfg_rate: Union[float, None] = None
    use_cuda_graph: bool = True
    use_flash_attention: bool = True
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16


# ─── 工具函数 ───

def _get_engine(request: Request):
    return request.app.state.inference_engine


def _task_error_response(task):
    if task.output.validation_error:
        return JSONResponse(status_code=400, content={"message": task.output.validation_error})
    return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": task.output.error})


async def tts_handle(req: dict, engine):
    """TTS 推理核心处理函数"""
    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")
    error = engine.validate_request(req)
    if error:
        return JSONResponse(status_code=400, content={"message": error})

    if streaming_mode == 0:
        streaming_mode = False
        return_fragment = False
        fixed_length_chunk = False
    elif streaming_mode == 1:
        streaming_mode = False
        return_fragment = True
        fixed_length_chunk = False
    elif streaming_mode == 2:
        streaming_mode = True
        return_fragment = False
        fixed_length_chunk = False
    elif streaming_mode == 3:
        streaming_mode = True
        return_fragment = False
        fixed_length_chunk = True
    else:
        return JSONResponse(
            status_code=400,
            content={"message": "the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)"},
        )

    req["streaming_mode"] = streaming_mode
    req["return_fragment"] = return_fragment
    req["fixed_length_chunk"] = fixed_length_chunk

    streaming_mode = streaming_mode or return_fragment
    task = None
    response_owns_task = False
    try:
        # Resolve the manual selection in queue order, independently of v3 voices.
        task = await engine.submit_selected(req, "raw" if streaming_mode and media_type == "wav" else media_type)
        if streaming_mode:
            while not task.output.done and not task.output.chunks:
                await task.output.wait_for_update()
            if task.output.error and not task.output.chunks:
                return _task_error_response(task)
            async def streaming_generator():
                index = 0
                try:
                    while not task.output.done or index < len(task.output.chunks):
                        while index < len(task.output.chunks):
                            chunk = task.output.chunks[index]
                            if index == 0 and media_type == "wav":
                                yield wave_header_chunk(sample_rate=chunk.sample_rate)
                            yield chunk.data
                            index += 1
                        if not task.output.done:
                            await task.output.wait_for_update()
                    if task.output.error:
                        raise RuntimeError(task.output.error)
                finally:
                    engine.cancel_task(task.task_id)
                    engine.cleanup_task(task.task_id)
            response_owns_task = True
            return StreamingResponse(
                streaming_generator(),
                media_type=f"audio/{media_type}",
            )
        # Legacy v2 waits through queueing and synthesis without a server deadline.
        await task.output.wait_until_done(timeout=None)
        if task.output.error:
            return _task_error_response(task)
        chunks = [chunk.data for chunk in task.output.chunks]
        audio_data = join_wav_chunks(chunks) if media_type == "wav" else b"".join(chunks)
        return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})
    finally:
        if task is not None and not response_owns_task:
            engine.cancel_task(task.task_id)
            engine.cleanup_task(task.task_id)


# ─── TTS 端点 ───

@router.get("/tts", summary="TTS 推理 (GET)")
async def tts_get_endpoint(
    request: Request,
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 15,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: Union[int, None] = None,
    cfg_rate: Union[float, None] = None,
    use_cuda_graph: bool = True,
    use_flash_attention: bool = True,
    super_sampling: bool = False,
    streaming_mode: Union[bool, int] = False,
    overlap_length: int = 2,
    min_chunk_length: int = 16,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower() if text_lang else text_lang,
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower() if prompt_lang else prompt_lang,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": sample_steps,
        "cfg_rate": cfg_rate,
        "use_cuda_graph": use_cuda_graph,
        "use_flash_attention": use_flash_attention,
        "super_sampling": super_sampling,
        "overlap_length": int(overlap_length),
        "min_chunk_length": int(min_chunk_length),
    }
    return await tts_handle(req, _get_engine(request))


@router.post("/tts", summary="TTS 推理 (POST)")
async def tts_post_endpoint(request: Request, body: TTS_Request):
    req = body.model_dump()
    return await tts_handle(req, _get_engine(request))


# ─── 模型切换端点 ───

@router.get("/set_gpt_weights", summary="切换 GPT 模型")
async def set_gpt_weights(request: Request, weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        await _get_engine(request).select_model_weights(gpt=weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@router.get("/set_sovits_weights", summary="切换 SoVITS 模型")
async def set_sovits_weights(request: Request, weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        await _get_engine(request).select_model_weights(sovits=weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})
