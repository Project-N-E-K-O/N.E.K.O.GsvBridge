"""
GPT-SoVITS API v3 - 配置管理路由

提供声音配置的 CRUD 接口。
"""

import asyncio
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api_neko.config import (
    PROJECT_ROOT,
    VoiceConfig,
    aload_voice,
    aload_default_config,
    alist_voices,
    asave_voice,
    reload_default_config,
    _dict_to_voice_config,
    afind_voice_file,
    aload_settings,
    asave_settings,
)


router = APIRouter(prefix="/api/v3", tags=["config"])


# ─── 响应 / 请求模型 ───


class VoiceInfoResponse(BaseModel):
    id: str
    name: str
    description: str


class ModelConfigResponse(BaseModel):
    gpt_weights: str
    sovits_weights: str
    version: str
    onnx_model_dir: str = ""
    backend: str = ""


class RefAudioResponse(BaseModel):
    path: str
    prompt_text: str
    prompt_lang: str
    aux_ref_audio_paths: list[str] = []


class InferParamsResponse(BaseModel):
    text_lang: str
    text_split_method: str
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    seed: int
    batch_size: int
    batch_threshold: float
    split_bucket: bool
    parallel_infer: bool
    speed_factor: float
    fragment_interval: float
    sample_steps: int
    super_sampling: bool
    streaming_mode: bool
    return_fragment: bool
    overlap_length: int
    min_chunk_length: int
    fixed_length_chunk: bool


class OutputConfigResponse(BaseModel):
    media_type: str


class VoiceConfigResponse(BaseModel):
    voice: VoiceInfoResponse
    model: ModelConfigResponse
    ref_audio: RefAudioResponse
    params: InferParamsResponse
    output: OutputConfigResponse


class VoiceListItem(BaseModel):
    id: str
    name: str
    description: str
    version: str


# VoiceConfigResponse 同时作为请求体（创建/更新）
VoiceConfigRequest = VoiceConfigResponse


class BatchDeleteRequest(BaseModel):
    ids: list[str]


# ─── 工具函数 ───


def _voice_config_to_response(cfg: VoiceConfig) -> dict:
    """将 VoiceConfig dataclass 转为可序列化的 dict"""
    return asdict(cfg)


# ─── 声音列表 ───


@router.get("/voices", response_model=list[VoiceListItem], summary="列出所有声音")
async def api_list_voices():
    """返回所有可用声音的摘要列表"""
    voices = await alist_voices()
    return [
        VoiceListItem(
            id=v.voice.id,
            name=v.voice.name,
            description=v.voice.description,
            version=v.model.version,
        )
        for v in voices
    ]


@router.get(
    "/voices/full",
    response_model=list[VoiceConfigResponse],
    summary="列出所有声音（完整配置）",
)
async def api_list_voices_full():
    """返回所有声音的完整配置，避免前端逐个请求"""
    voices = await alist_voices()
    return [_voice_config_to_response(v) for v in voices]


# ─── 声音 CRUD ───


@router.post(
    "/voices",
    response_model=VoiceConfigResponse,
    status_code=201,
    summary="创建声音配置",
)
async def api_create_voice(body: VoiceConfigRequest):
    """创建新的声音配置文件"""
    voice_id = body.voice.id
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice.id 不能为空")

    # 检查是否已存在
    existing = await afind_voice_file(voice_id)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"声音 {voice_id!r} 已存在")

    cfg = _dict_to_voice_config(body.model_dump())
    try:
        await asave_voice(cfg)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _voice_config_to_response(cfg)


@router.post("/voices/batch-delete", summary="批量删除声音配置")
async def api_batch_delete_voices(body: BatchDeleteRequest):
    """批量删除多个声音配置"""
    deleted = []
    not_found = []
    for vid in body.ids:
        file_path = await afind_voice_file(vid)
        if file_path is None:
            not_found.append(vid)
        else:
            file_path.unlink()
            deleted.append(vid)
    return {"status": "ok", "deleted": deleted, "not_found": not_found}


@router.get(
    "/voices/{voice_id}", response_model=VoiceConfigResponse, summary="获取声音配置"
)
async def api_get_voice(voice_id: str):
    """获取指定声音的完整配置（已合并默认配置）"""
    try:
        cfg = await aload_voice(voice_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"声音 {voice_id!r} 不存在")
    return _voice_config_to_response(cfg)


@router.put(
    "/voices/{voice_id}", response_model=VoiceConfigResponse, summary="更新声音配置"
)
async def api_update_voice(voice_id: str, body: VoiceConfigRequest):
    """更新已有的声音配置"""
    existing = await afind_voice_file(voice_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"声音 {voice_id!r} 不存在")

    cfg = _dict_to_voice_config(body.model_dump())
    # 确保 voice.id 与路径参数一致
    cfg.voice.id = voice_id
    try:
        await asave_voice(cfg, path=existing)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _voice_config_to_response(cfg)


@router.delete("/voices/{voice_id}", summary="删除声音配置")
async def api_delete_voice(voice_id: str):
    """删除指定声音的配置文件"""
    file_path = await afind_voice_file(voice_id)
    if file_path is None:
        raise HTTPException(status_code=404, detail=f"声音 {voice_id!r} 不存在")

    file_path.unlink()
    return {"status": "ok", "deleted": voice_id}


# ─── 默认配置 ───


@router.get(
    "/config/default", response_model=VoiceConfigResponse, summary="获取默认配置"
)
async def api_get_default_config():
    """获取默认配置（default.toml 的内容）"""
    cfg = await aload_default_config()
    return _voice_config_to_response(cfg)


@router.post("/config/reload", summary="重新加载配置")
async def api_reload_config():
    """重新加载默认配置缓存（修改 default.toml 后调用）"""
    reload_default_config()
    return {"status": "ok", "message": "默认配置已重新加载"}


# ─── 文件扫描 ───

_PROJECT_ROOT = PROJECT_ROOT

# 权重文件夹映射（version → 文件夹名）
_GPT_WEIGHT_ROOTS = {
    "v1": "GPT_weights",
    "v2": "GPT_weights_v2",
    "v3": "GPT_weights_v3",
    "v4": "GPT_weights_v4",
    "v2Pro": "GPT_weights_v2Pro",
    "v2ProPlus": "GPT_weights_v2ProPlus",
}
_SOVITS_WEIGHT_ROOTS = {
    "v1": "SoVITS_weights",
    "v2": "SoVITS_weights_v2",
    "v3": "SoVITS_weights_v3",
    "v4": "SoVITS_weights_v4",
    "v2Pro": "SoVITS_weights_v2Pro",
    "v2ProPlus": "SoVITS_weights_v2ProPlus",
}

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# 预训练模型目录
_PRETRAINED_DIR = "GPT_SoVITS/pretrained_models"
# 推理时不需要的文件名模式（鉴别器、vocoder 等）
_NON_INFERENCE_PREFIXES = ("s2D",)
_NON_INFERENCE_NAMES = {"vocoder.pth"}


def _scan_weights(
    root_map: dict, extensions: set, version: Optional[str] = None
) -> list[str]:
    """扫描权重文件，返回相对于项目根目录的路径列表（含 pretrained_models）"""
    results = []
    dirs = {version: root_map[version]} if version and version in root_map else root_map
    for ver, dirname in dirs.items():
        folder = _PROJECT_ROOT / dirname
        if not folder.is_dir():
            continue
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in extensions:
                results.append(str(f.relative_to(_PROJECT_ROOT)))

    # 追加扫描 pretrained_models 目录
    pretrained = _PROJECT_ROOT / _PRETRAINED_DIR
    if pretrained.is_dir():
        for f in sorted(pretrained.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in extensions:
                continue
            # 排除非推理用文件（鉴别器 s2D*.pth、vocoder.pth 等）
            if f.name in _NON_INFERENCE_NAMES:
                continue
            if any(f.name.startswith(p) for p in _NON_INFERENCE_PREFIXES):
                continue
            rel = str(f.relative_to(_PROJECT_ROOT))
            if rel not in results:
                results.append(rel)
    return results


@router.get("/scan/gpt-weights", summary="扫描 GPT 权重文件")
async def api_scan_gpt_weights(
    version: Optional[str] = Query(None, description="按版本过滤"),
):
    """扫描 GPT_weights* 目录下的 .ckpt 文件"""
    files = await asyncio.to_thread(
        _scan_weights, _GPT_WEIGHT_ROOTS, {".ckpt"}, version
    )
    return {"files": files}


@router.get("/scan/sovits-weights", summary="扫描 SoVITS 权重文件")
async def api_scan_sovits_weights(
    version: Optional[str] = Query(None, description="按版本过滤"),
):
    """扫描 SoVITS_weights* 目录下的 .pth 文件"""
    files = await asyncio.to_thread(
        _scan_weights, _SOVITS_WEIGHT_ROOTS, {".pth"}, version
    )
    return {"files": files}


# 音频搜索根目录（与版本目录对应，训练工具标准输出结构）
_AUDIO_SEARCH_ROOTS = [
    "v1",
    "v2",
    "v3",
    "v4",
    "v2Pro",
    "v2ProPlus",
    "voices",
    "api_neko/voices",
    "reference_audios",
]


@router.get("/scan/audio", summary="扫描音频文件")
async def api_scan_audio(
    dir: Optional[str] = Query(
        None, description="扫描目录（相对于项目根），为空则自动搜索预定义目录"
    ),
):
    """扫描音频文件。不指定 dir 时自动搜索预定义的版本目录。"""

    def _scan():
        results = []
        if dir:
            targets = [(_PROJECT_ROOT / dir).resolve()]
        else:
            targets = [_PROJECT_ROOT / d for d in _AUDIO_SEARCH_ROOTS]

        for target in targets:
            if not target.is_dir():
                continue
            # 安全检查
            if not str(target.resolve()).startswith(str(_PROJECT_ROOT)):
                continue
            for f in sorted(target.rglob("*")):
                if f.is_file() and f.suffix.lower() in _AUDIO_EXTENSIONS:
                    results.append(str(f.relative_to(_PROJECT_ROOT)))
        return results

    files = await asyncio.to_thread(_scan)
    return {"files": files}


# ─── 用户设置 ───


@router.get("/settings", summary="获取用户设置")
async def api_get_settings():
    """读取 settings.toml"""
    return await aload_settings()


@router.put("/settings", summary="保存用户设置")
async def api_put_settings(data: dict):
    """写入 settings.toml"""
    await asave_settings(data)
    return {"status": "ok"}


# ─── 模型管理 ───

# Genie ONNX 模型必需文件
_ONNX_REQUIRED_FILES = {
    "t2s_encoder_fp32.onnx",
    "t2s_first_stage_decoder_fp32.onnx",
    "t2s_stage_decoder_fp32.onnx",
    "vits_fp32.onnx",
}
_ONNX_OPTIONAL_FILES = {
    "prompt_encoder_fp32.onnx",  # V2ProPlus
    "t2s_shared_fp16.bin",
    "vits_fp16.bin",
    "prompt_encoder_fp16.bin",
}

# Genie ONNX 模型搜索根目录（可配置）
_ONNX_SEARCH_ROOTS = ["onnx_models", "Genie-TTS/models", "models"]


def _parse_gsv_filename(filename: str) -> dict:
    """从 GSV 权重文件名解析元数据（角色名、epoch、step）"""
    stem = Path(filename).stem
    info = {"character": "", "epoch": None, "step": None}

    # 匹配 epoch: xxx-e10 或 xxx_e10
    m = re.search(r"[-_]e(\d+)", stem)
    if m:
        info["epoch"] = int(m.group(1))

    # 匹配 step: xxx_s230 或 xxx-s230
    m = re.search(r"[-_]s(\d+)", stem)
    if m:
        info["step"] = int(m.group(1))

    # 角色名：去掉 epoch/step 后缀
    name = re.sub(r"[-_][es]\d+", "", stem).strip("-_ ")
    info["character"] = name
    return info


def _file_meta(filepath: Path) -> dict:
    """获取文件元数据"""
    stat = filepath.stat()
    return {
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _dir_size(dirpath: Path) -> int:
    """计算目录总大小"""
    total = 0
    for f in dirpath.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


# pretrained 子目录 → version 映射
_PRETRAINED_SUBDIR_VERSION = {
    "gsv-v2final-pretrained": "v2",
    "gsv-v4-pretrained": "v4",
    "v2Pro": "v2Pro",
}


def _infer_pretrained_version(filepath: Path) -> str:
    """根据文件所在子目录或文件名推断 pretrained 模型的版本"""
    # 先看子目录名
    parent_name = filepath.parent.name
    if parent_name in _PRETRAINED_SUBDIR_VERSION:
        return _PRETRAINED_SUBDIR_VERSION[parent_name]
    # 根目录下按文件名判断
    name = filepath.stem.lower()
    if "v4" in name:
        return "v4"
    if "v3" in name:
        return "v3"
    if "v2pro" in name:
        return "v2Pro"
    # 默认 v2（如 s1bert25hz-2kh-longer, s2G488k 等都是 v1/v2 pretrained）
    return "v2"


def _scan_pretrained_models(
    gpt_files: dict[str, list[dict]],
    sovits_files: dict[str, list[dict]],
) -> None:
    """扫描 GPT_SoVITS/pretrained_models/ 下的预训练模型，追加到 gpt_files/sovits_files"""
    pretrained = _PROJECT_ROOT / _PRETRAINED_DIR
    if not pretrained.is_dir():
        return

    for f in sorted(pretrained.rglob("*")):
        if not f.is_file():
            continue
        # 排除非推理用文件
        if f.name in _NON_INFERENCE_NAMES:
            continue
        if any(f.name.startswith(p) for p in _NON_INFERENCE_PREFIXES):
            continue

        version = _infer_pretrained_version(f)
        parsed = _parse_gsv_filename(f.name)
        meta = _file_meta(f)
        rel_path = str(f.relative_to(_PROJECT_ROOT))
        entry = {
            "path": rel_path,
            "filename": f.name,
            "version": version,
            "character": parsed["character"],
            "epoch": parsed["epoch"],
            "step": parsed["step"],
            "pretrained": True,
            **meta,
        }

        if f.suffix.lower() == ".ckpt":
            gpt_files.setdefault(version, []).append(entry)
        elif f.suffix.lower() == ".pth":
            sovits_files.setdefault(version, []).append(entry)


def _scan_gsv_models() -> list[dict]:
    """扫描 GSV PyTorch 模型（含 pretrained_models），按角色配对 GPT+SoVITS"""
    # 先收集所有 GPT 和 SoVITS 文件
    gpt_files: dict[str, list[dict]] = {}  # version → [file_info]
    sovits_files: dict[str, list[dict]] = {}

    for version, dirname in _GPT_WEIGHT_ROOTS.items():
        folder = _PROJECT_ROOT / dirname
        if not folder.is_dir():
            continue
        for f in sorted(folder.rglob("*.ckpt")):
            if not f.is_file():
                continue
            parsed = _parse_gsv_filename(f.name)
            meta = _file_meta(f)
            rel_path = str(f.relative_to(_PROJECT_ROOT))
            entry = {
                "path": rel_path,
                "filename": f.name,
                "version": version,
                "character": parsed["character"],
                "epoch": parsed["epoch"],
                "step": parsed["step"],
                "pretrained": False,
                **meta,
            }
            gpt_files.setdefault(version, []).append(entry)

    for version, dirname in _SOVITS_WEIGHT_ROOTS.items():
        folder = _PROJECT_ROOT / dirname
        if not folder.is_dir():
            continue
        for f in sorted(folder.rglob("*.pth")):
            if not f.is_file():
                continue
            parsed = _parse_gsv_filename(f.name)
            meta = _file_meta(f)
            rel_path = str(f.relative_to(_PROJECT_ROOT))
            entry = {
                "path": rel_path,
                "filename": f.name,
                "version": version,
                "character": parsed["character"],
                "epoch": parsed["epoch"],
                "step": parsed["step"],
                "pretrained": False,
                **meta,
            }
            sovits_files.setdefault(version, []).append(entry)

    # ─── 追加扫描 pretrained_models ───
    _scan_pretrained_models(gpt_files, sovits_files)

    # 构建模型列表：每个 GPT 文件作为一个模型条目
    models = []
    for version, gpt_list in gpt_files.items():
        for gpt in gpt_list:
            # 尝试匹配同角色的 SoVITS
            matched_sovits = None
            for sv in sovits_files.get(version, []):
                if sv["character"] == gpt["character"]:
                    matched_sovits = sv
                    break

            total_size = gpt["size_bytes"]
            if matched_sovits:
                total_size += matched_sovits["size_bytes"]

            models.append(
                {
                    "type": "pytorch",
                    "pretrained": gpt.get("pretrained", False),
                    "version": version,
                    "character": gpt["character"],
                    "total_size_bytes": total_size,
                    "modified": gpt["modified"],
                    "gpt": gpt,
                    "sovits": matched_sovits,
                }
            )

    # 添加没有配对 GPT 的 SoVITS 文件
    gpt_chars = {
        (g["version"], g["character"]) for gl in gpt_files.values() for g in gl
    }
    for version, sv_list in sovits_files.items():
        for sv in sv_list:
            if (version, sv["character"]) not in gpt_chars:
                models.append(
                    {
                        "type": "pytorch",
                        "pretrained": sv.get("pretrained", False),
                        "version": version,
                        "character": sv["character"],
                        "total_size_bytes": sv["size_bytes"],
                        "modified": sv["modified"],
                        "gpt": None,
                        "sovits": sv,
                    }
                )

    return models


def _scan_onnx_models() -> list[dict]:
    """扫描 Genie ONNX 模型目录"""
    models = []
    searched = set()

    for root_name in _ONNX_SEARCH_ROOTS:
        root = _PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        # 每个子目录可能是一个角色模型
        for d in sorted(root.iterdir()):
            if not d.is_dir() or str(d) in searched:
                continue
            searched.add(str(d))

            existing = {f.name for f in d.iterdir() if f.is_file()}
            has_required = _ONNX_REQUIRED_FILES.issubset(existing)
            if not has_required and not existing.intersection(_ONNX_REQUIRED_FILES):
                continue  # 不是 ONNX 模型目录

            missing = _ONNX_REQUIRED_FILES - existing
            has_prompt_encoder = "prompt_encoder_fp32.onnx" in existing
            has_fp16 = bool(
                existing.intersection({"t2s_shared_fp16.bin", "vits_fp16.bin"})
            )

            total_size = _dir_size(d)
            # 最新修改时间
            latest_mtime = max(
                (f.stat().st_mtime for f in d.rglob("*") if f.is_file()),
                default=0,
            )

            models.append(
                {
                    "type": "onnx",
                    "version": "V2ProPlus" if has_prompt_encoder else "V2",
                    "character": d.name,
                    "total_size_bytes": total_size,
                    "modified": datetime.fromtimestamp(latest_mtime).isoformat()
                    if latest_mtime
                    else "",
                    "onnx_model_dir": str(d.relative_to(_PROJECT_ROOT)),
                    "complete": len(missing) == 0,
                    "missing_files": sorted(missing) if missing else [],
                    "has_fp16": has_fp16,
                    "has_prompt_encoder": has_prompt_encoder,
                    "file_count": len(existing),
                }
            )

    return models


def _get_voice_references() -> dict[str, str]:
    """获取所有 voice 配置中引用的模型路径 → voice_id 映射"""
    from api_neko._toml_compat import load_toml

    refs = {}
    voices_dir = _PROJECT_ROOT / "api_neko" / "voices"
    if not voices_dir.is_dir():
        return refs
    for f in voices_dir.glob("*.toml"):
        if f.name.startswith("_"):
            continue
        try:
            data = load_toml(f)
            voice_id = data.get("voice", {}).get("id", f.stem)
            model = data.get("model", {})
            if model.get("gpt_weights"):
                refs[model["gpt_weights"]] = voice_id
            if model.get("sovits_weights"):
                refs[model["sovits_weights"]] = voice_id
            if model.get("onnx_model_dir"):
                refs[model["onnx_model_dir"]] = voice_id
        except Exception:
            continue
    return refs


@router.get("/models", summary="扫描所有模型")
async def api_list_models(
    type: Optional[str] = Query(None, description="过滤类型: pytorch | onnx"),
    version: Optional[str] = Query(
        None, description="过滤版本: v1, v2, v3, v4, v2Pro, v2ProPlus, V2, V2ProPlus"
    ),
):
    """统一扫描 GSV PyTorch 和 Genie ONNX 模型，返回结构化元数据"""

    def _scan():
        models = []
        voice_refs = _get_voice_references()

        if type != "onnx":
            for m in _scan_gsv_models():
                # 标记是否被 voice 引用
                gpt_path = m["gpt"]["path"] if m["gpt"] else ""
                sv_path = m["sovits"]["path"] if m["sovits"] else ""
                m["voice_id"] = (
                    voice_refs.get(gpt_path) or voice_refs.get(sv_path) or ""
                )
                models.append(m)

        if type != "pytorch":
            for m in _scan_onnx_models():
                m["voice_id"] = voice_refs.get(m.get("onnx_model_dir", "")) or ""
                models.append(m)

        # 版本过滤
        if version:
            v_lower = version.lower()
            models = [m for m in models if m["version"].lower() == v_lower]

        return models

    models = await asyncio.to_thread(_scan)
    return {"models": models, "total": len(models)}


@router.get("/models/scan/onnx", summary="扫描 ONNX 模型目录列表")
async def api_scan_onnx_dirs():
    """返回所有可用的 ONNX 模型目录（相对路径），供配置页选择"""

    def _scan():
        dirs = []
        searched = set()
        for root_name in _ONNX_SEARCH_ROOTS:
            root = _PROJECT_ROOT / root_name
            if not root.is_dir():
                continue
            real = root.resolve()
            if real in searched:
                continue
            searched.add(real)
            for d in sorted(root.iterdir()):
                if not d.is_dir():
                    continue
                onnx_files = list(d.glob("*.onnx"))
                if not onnx_files:
                    continue
                rel = str(d.relative_to(_PROJECT_ROOT))
                dirs.append(rel)
        return dirs

    dirs = await asyncio.to_thread(_scan)
    return {"dirs": dirs}


# ─── 模型转换 ───


class ConvertStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class _ConvertTask:
    """单个转换任务的状态"""

    __slots__ = (
        "id",
        "status",
        "progress",
        "message",
        "gpt_weights",
        "sovits_weights",
        "output_dir",
        "created_at",
        "finished_at",
    )

    def __init__(
        self, task_id: str, gpt_weights: str, sovits_weights: str, output_dir: str
    ):
        self.id = task_id
        self.status = ConvertStatus.PENDING
        self.progress = 0  # 0~100
        self.message = ""
        self.gpt_weights = gpt_weights
        self.sovits_weights = sovits_weights
        self.output_dir = output_dir
        self.created_at = time.time()
        self.finished_at: float = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "gpt_weights": self.gpt_weights,
            "sovits_weights": self.sovits_weights,
            "output_dir": self.output_dir,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


# 全局任务存储（内存，重启丢失）
_convert_tasks: dict[str, _ConvertTask] = {}
_convert_lock = threading.Lock()


def _run_convert(task: _ConvertTask):
    """在后台线程中执行模型转换"""
    task.status = ConvertStatus.RUNNING
    task.progress = 5
    task.message = "正在初始化转换器..."

    try:
        # 确保 Genie-TTS src 在 sys.path 中
        genie_src = str(_PROJECT_ROOT / "Genie-TTS" / "src")
        if genie_src not in sys.path:
            sys.path.insert(0, genie_src)

        gpt_path = (
            str(_PROJECT_ROOT / task.gpt_weights)
            if not os.path.isabs(task.gpt_weights)
            else task.gpt_weights
        )
        sovits_path = (
            str(_PROJECT_ROOT / task.sovits_weights)
            if not os.path.isabs(task.sovits_weights)
            else task.sovits_weights
        )
        out_dir = (
            str(_PROJECT_ROOT / task.output_dir)
            if not os.path.isabs(task.output_dir)
            else task.output_dir
        )

        # 检查输入文件
        if not os.path.isfile(gpt_path):
            raise FileNotFoundError(f"GPT 权重文件不存在: {gpt_path}")
        if not os.path.isfile(sovits_path):
            raise FileNotFoundError(f"SoVITS 权重文件不存在: {sovits_path}")

        os.makedirs(out_dir, exist_ok=True)

        task.progress = 10
        task.message = "正在加载转换模块..."

        # 绕过 genie_tts/__init__.py 的完整导入链（会触发交互式 input 和资源检查）
        # 只需要 Converter 子包，不需要完整运行时
        import types
        import builtins

        # 1. 创建占位 GenieData 目录（避免 Resources.py 的 input() 阻塞）
        genie_data_dir = str(_PROJECT_ROOT / "GenieData")
        os.environ["GENIE_DATA_DIR"] = genie_data_dir
        os.makedirs(genie_data_dir, exist_ok=True)
        # 创建 Resources.py 检查的必需目录/文件占位
        hubert_dir = os.path.join(genie_data_dir, "chinese-hubert-base")
        sv_model = os.path.join(genie_data_dir, "speaker_encoder.onnx")
        os.makedirs(hubert_dir, exist_ok=True)
        if not os.path.exists(sv_model):
            open(sv_model, "w").close()

        # 2. 临时 mock input() 防止阻塞
        original_input = builtins.input
        builtins.input = lambda *a, **kw: "n"
        try:
            from genie_tts.Converter.Converter import convert
        finally:
            builtins.input = original_input

        task.progress = 20
        task.message = "正在转换模型（这可能需要几分钟）..."

        convert(
            torch_ckpt_path=gpt_path,
            torch_pth_path=sovits_path,
            output_dir=out_dir,
        )

        task.progress = 100
        task.status = ConvertStatus.DONE
        task.message = f"转换完成，输出目录: {out_dir}"

    except Exception as e:
        task.status = ConvertStatus.ERROR
        task.message = str(e)
    finally:
        task.finished_at = time.time()


class ConvertRequest(BaseModel):
    gpt_weights: str  # .ckpt 路径（相对于项目根或绝对路径）
    sovits_weights: str  # .pth 路径
    output_dir: str  # ONNX 输出目录


@router.post("/models/convert", summary="提交模型转换任务")
async def api_convert_model(body: ConvertRequest):
    """将 PyTorch 模型转换为 Genie ONNX 格式（后台异步执行）"""
    task_id = uuid.uuid4().hex[:12]
    task = _ConvertTask(task_id, body.gpt_weights, body.sovits_weights, body.output_dir)

    with _convert_lock:
        # 检查是否有正在运行的转换任务
        running = [
            t for t in _convert_tasks.values() if t.status == ConvertStatus.RUNNING
        ]
        if running:
            return {
                "error": "已有转换任务正在运行，请等待完成后再提交",
                "running_task_id": running[0].id,
            }
        _convert_tasks[task_id] = task

    # 启动后台线程
    thread = threading.Thread(target=_run_convert, args=(task,), daemon=True)
    thread.start()

    return {"task_id": task_id, "status": task.status.value}


@router.get("/models/convert/{task_id}", summary="查询转换任务状态")
async def api_convert_status(task_id: str):
    """查询模型转换任务的当前状态"""
    task = _convert_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"转换任务 {task_id} 不存在")
    return task.to_dict()


@router.get("/models/convert", summary="列出所有转换任务")
async def api_list_convert_tasks():
    """列出所有转换任务（含历史）"""
    return {"tasks": [t.to_dict() for t in _convert_tasks.values()]}
