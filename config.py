"""
GPT-SoVITS API v3 - 声音配置管理模块

使用 TOML 文件存储每个声音角色的推理参数。
每个 voice_id 对应 voices/ 目录下的一个 .toml 文件。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from api_neko._toml_compat import load_toml, dump_toml


# ─── 路径配置 ───

_api_neko_DIR = Path(__file__).resolve().parent
VOICES_DIR = _api_neko_DIR / "voices"


def _resolve_project_root() -> Path:
    """从 settings.toml 读取 project_root，留空则自动推断为 api_neko 的父目录"""
    try:
        settings = load_toml(_api_neko_DIR / "settings.toml")
        root = settings.get("project_root", "")
        if root:
            return Path(root).resolve()
    except Exception:
        pass
    return _api_neko_DIR.parent


def _resolve_gsv_package_dir(project_root: Path) -> Path:
    """从 settings.toml 读取 gsv_package_dir，留空则推断为 project_root/GPT_SoVITS"""
    try:
        settings = load_toml(_api_neko_DIR / "settings.toml")
        d = settings.get("gsv_package_dir", "")
        if d:
            return Path(d).resolve()
    except Exception:
        pass
    return project_root / "GPT_SoVITS"


PROJECT_ROOT: Path = _resolve_project_root()
GSV_PACKAGE_DIR: Path = _resolve_gsv_package_dir(PROJECT_ROOT)


@dataclass
class VoiceInfo:
    """声音基本信息"""
    id: str = ""
    name: str = ""
    description: str = ""


@dataclass
class ModelConfig:
    """模型权重配置"""
    gpt_weights: str = ""
    sovits_weights: str = ""
    version: str = "v2"
    onnx_model_dir: str = ""  # Genie 后端: ONNX 模型目录路径
    backend: str = ""         # 后端类型: "gsv" | "genie"，留空则使用 settings.toml 全局配置


@dataclass
class RefAudioConfig:
    """参考音频配置"""
    path: str = ""
    prompt_text: str = ""
    prompt_lang: str = "all_zh"
    aux_ref_audio_paths: list[str] = field(default_factory=list)


@dataclass
class InferParams:
    """推理参数"""
    text_lang: str = "all_zh"
    text_split_method: str = "cut5"
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    seed: int = -1
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    parallel_infer: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    sample_steps: int = 32
    super_sampling: bool = False
    streaming_mode: bool = False
    return_fragment: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16
    fixed_length_chunk: bool = False


@dataclass
class OutputConfig:
    """输出配置"""
    media_type: str = "wav"


@dataclass
class VoiceConfig:
    """完整的声音配置"""
    voice: VoiceInfo = field(default_factory=VoiceInfo)
    model: ModelConfig = field(default_factory=ModelConfig)
    ref_audio: RefAudioConfig = field(default_factory=RefAudioConfig)
    params: InferParams = field(default_factory=InferParams)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_tts_inputs(self, text: str, **overrides) -> dict:
        """
        将配置转换为 TTS.run() 所需的 inputs dict。
        overrides 中的参数会覆盖配置文件中的默认值。
        """
        inputs = {
            "text": text,
            "text_lang": overrides.get("text_lang", self.params.text_lang),
            "ref_audio_path": self._resolve_path(self.ref_audio.path),
            "aux_ref_audio_paths": [
                self._resolve_path(p) for p in self.ref_audio.aux_ref_audio_paths
            ],
            "prompt_text": overrides.get("prompt_text", self.ref_audio.prompt_text),
            "prompt_lang": overrides.get("prompt_lang", self.ref_audio.prompt_lang),
            "top_k": overrides.get("top_k", self.params.top_k),
            "top_p": overrides.get("top_p", self.params.top_p),
            "temperature": overrides.get("temperature", self.params.temperature),
            "text_split_method": overrides.get("text_split_method", self.params.text_split_method),
            "batch_size": overrides.get("batch_size", self.params.batch_size),
            "batch_threshold": overrides.get("batch_threshold", self.params.batch_threshold),
            "speed_factor": overrides.get("speed_factor", self.params.speed_factor),
            "split_bucket": overrides.get("split_bucket", self.params.split_bucket),
            "return_fragment": overrides.get("return_fragment", self.params.return_fragment),
            "fragment_interval": overrides.get("fragment_interval", self.params.fragment_interval),
            "seed": overrides.get("seed", self.params.seed),
            "parallel_infer": overrides.get("parallel_infer", self.params.parallel_infer),
            "repetition_penalty": overrides.get("repetition_penalty", self.params.repetition_penalty),
            "sample_steps": overrides.get("sample_steps", self.params.sample_steps),
            "super_sampling": overrides.get("super_sampling", self.params.super_sampling),
            "streaming_mode": overrides.get("streaming_mode", self.params.streaming_mode),
            "overlap_length": overrides.get("overlap_length", self.params.overlap_length),
            "min_chunk_length": overrides.get("min_chunk_length", self.params.min_chunk_length),
            "fixed_length_chunk": overrides.get("fixed_length_chunk", self.params.fixed_length_chunk),
        }
        return inputs

    def _resolve_path(self, path: str) -> str:
        """将相对路径解析为绝对路径（相对于项目根目录）"""
        if not path:
            return ""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(PROJECT_ROOT / p)


# ─── 内部工具 ───

def _load_raw_toml(path: str | Path) -> dict:
    """加载 TOML 文件为原始 dict"""
    return load_toml(path)


def _deep_merge(base: dict, override: dict) -> dict:
    """
    深度合并两个 dict。override 中的值会覆盖 base 中的值。
    对于嵌套 dict 会递归合并，非 dict 值直接覆盖。
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_voice_config(data: dict) -> VoiceConfig:
    """将原始 dict 转换为 VoiceConfig dataclass"""
    cfg = VoiceConfig()

    if "voice" in data:
        v = data["voice"]
        cfg.voice = VoiceInfo(
            id=v.get("id", ""),
            name=v.get("name", ""),
            description=v.get("description", ""),
        )

    if "model" in data:
        m = data["model"]
        cfg.model = ModelConfig(
            gpt_weights=m.get("gpt_weights", ""),
            sovits_weights=m.get("sovits_weights", ""),
            version=m.get("version", "v2"),
            onnx_model_dir=m.get("onnx_model_dir", ""),
            backend=m.get("backend", ""),
        )

    if "ref_audio" in data:
        r = data["ref_audio"]
        cfg.ref_audio = RefAudioConfig(
            path=r.get("path", ""),
            prompt_text=r.get("prompt_text", ""),
            prompt_lang=r.get("prompt_lang", "all_zh"),
            aux_ref_audio_paths=r.get("aux_ref_audio_paths", []),
        )

    if "params" in data:
        p = data["params"]
        defaults = InferParams()
        cfg.params = InferParams(**{
            k: p.get(k, getattr(defaults, k))
            for k in defaults.__dataclass_fields__
        })

    if "output" in data:
        o = data["output"]
        cfg.output = OutputConfig(
            media_type=o.get("media_type", "wav"),
        )

    return cfg


# ─── 默认配置 ───

_default_data_cache: dict | None = None


def _get_default_data() -> dict:
    """加载并缓存 default.toml 的原始数据"""
    global _default_data_cache
    if _default_data_cache is None:
        default_path = VOICES_DIR / "default.toml"
        if default_path.exists():
            _default_data_cache = _load_raw_toml(default_path)
        else:
            _default_data_cache = {}
    return _default_data_cache


def reload_default_config():
    """强制重新加载默认配置（修改 default.toml 后调用）"""
    global _default_data_cache
    _default_data_cache = None


def load_default_config() -> VoiceConfig:
    """加载默认配置"""
    return _dict_to_voice_config(_get_default_data())


async def aload_default_config() -> VoiceConfig:
    """load_default_config 的异步版本"""
    return await asyncio.to_thread(load_default_config)


# ─── TOML 读写 ───

def load_voice(voice_id: str) -> VoiceConfig:
    """根据 voice_id 加载对应的 TOML 配置文件，自动合并默认配置"""
    toml_path = _find_voice_file(voice_id)
    if toml_path is None:
        raise FileNotFoundError(f"未找到 voice_id={voice_id!r} 的配置文件")
    return load_voice_from_file(toml_path)


async def aload_voice(voice_id: str) -> VoiceConfig:
    """load_voice 的异步版本"""
    return await asyncio.to_thread(load_voice, voice_id)


def load_voice_from_file(path: str | Path, apply_defaults: bool = True) -> VoiceConfig:
    """
    从指定路径加载 TOML 配置。
    当 apply_defaults=True 时，先加载 default.toml 作为基础，再用声音配置覆盖。
    """
    voice_data = _load_raw_toml(path)

    if apply_defaults:
        default_data = _get_default_data()
        merged = _deep_merge(default_data, voice_data)
    else:
        merged = voice_data

    return _dict_to_voice_config(merged)


def save_voice(config: VoiceConfig, path: Optional[str | Path] = None) -> Path:
    """
    保存声音配置到 TOML 文件。
    如果未指定 path，则保存到 voices/{voice_id}.toml
    """
    if path is None:
        if not config.voice.id:
            raise ValueError("voice.id 不能为空")
        path = VOICES_DIR / f"{config.voice.id}.toml"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(config.ref_audio)
    data = {
        "voice": asdict(config.voice),
        "model": asdict(config.model),
        "ref_audio": asdict(config.ref_audio),
        "params": asdict(config.params),
        "output": asdict(config.output),
    }

    a = str(Path(config.ref_audio.path))
    b = str(Path(config.model.gpt_weights))
    c = str(Path(config.model.sovits_weights))
    aa = a.replace("\\","/")
    bb = b.replace("\\","/")
    cc = c.replace("\\","/")
    data['ref_audio']['path'] = aa
    data['model']['gpt_weights'] = bb
    data['model']['sovits_weights'] = cc
    config.ref_audio.path = aa
    config.model.gpt_weights = bb
    config.model.sovits_weights = cc
    dump_toml(data, path)

    return path


async def asave_voice(config: VoiceConfig, path: Optional[str | Path] = None) -> Path:
    """save_voice 的异步版本"""
    return await asyncio.to_thread(save_voice, config, path)


def list_voices() -> list[VoiceConfig]:
    """列出所有可用的声音配置"""
    voices = []
    if not VOICES_DIR.exists():
        return voices
    for toml_file in sorted(VOICES_DIR.glob("[!_]*.toml")):
        if toml_file.name == "default.toml":
            continue
        try:
            cfg = load_voice_from_file(toml_file)
            voices.append(cfg)
        except Exception as e:
            print(f"[warn] 加载 {toml_file.name} 失败: {e}")
    return voices


async def alist_voices() -> list[VoiceConfig]:
    """list_voices 的异步版本"""
    return await asyncio.to_thread(list_voices)


def _find_voice_file(voice_id: str) -> Optional[Path]:
    """
    查找 voice_id 对应的 TOML 文件。
    优先匹配文件名，其次匹配文件内 voice.id 字段。
    """
    # 1. 直接文件名匹配
    direct = VOICES_DIR / f"{voice_id}.toml"
    if direct.exists():
        return direct

    # 2. 遍历匹配 voice.id 字段
    if VOICES_DIR.exists():
        for toml_file in VOICES_DIR.glob("*.toml"):
            try:
                data = load_toml(toml_file)
                if data.get("voice", {}).get("id") == voice_id:
                    return toml_file
            except Exception:
                continue

    return None


async def afind_voice_file(voice_id: str) -> Optional[Path]:
    """_find_voice_file 的异步版本"""
    return await asyncio.to_thread(_find_voice_file, voice_id)


# ─── 用户设置 ───

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.toml"


def load_settings() -> dict:
    """加载用户设置，不存在则返回默认值"""
    if SETTINGS_PATH.exists():
        try:
            return _load_raw_toml(SETTINGS_PATH)
        except Exception:
            pass
    return {"last_voice_id": "_default"}


def save_settings(data: dict) -> None:
    """保存用户设置到 settings.toml（合并写入，不丢失已有字段）"""
    existing = load_settings()
    existing.update(data)
    dump_toml(existing, SETTINGS_PATH)


async def aload_settings() -> dict:
    return await asyncio.to_thread(load_settings)


async def asave_settings(data: dict) -> None:
    return await asyncio.to_thread(save_settings, data)
