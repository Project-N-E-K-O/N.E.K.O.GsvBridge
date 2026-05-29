"""
Genie-TTS ONNX 轻量推理后端

适配 Genie-TTS 的 ONNX 推理引擎，实现 BasePipeline 接口。
支持按句分段的流式输出以降低首音延迟。

依赖：genie_tts 包（需要在 sys.path 中包含 Genie-TTS/src）
"""

import builtins
import logging
import os
import sys
from pathlib import Path
from typing import Generator, Tuple, Optional, List

import numpy as np

from .base import BasePipeline, BackendInfo

logger = logging.getLogger(__name__)

# Genie-TTS 语言映射：api_neko 格式 → Genie 格式
_LANG_MAP = {
    "all_zh": "chinese",
    "all_ja": "japanese",
    "all_ko": "korean",
    "all_en": "english",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "en": "english",
    "chinese": "chinese",
    "japanese": "japanese",
    "korean": "korean",
    "english": "english",
    # 混合语言
    "zh_en": "hybrid-chinese-english",
    "all_mix": "hybrid-chinese-english",
}

# Genie 支持的语言列表（用于参数校验）
_SUPPORTED_LANGUAGES = list(set(_LANG_MAP.keys()))


def _ensure_genie_importable():
    """确保 genie_tts 包可导入，处理 Resources.py 的交互式检查"""
    from api_neko.config import PROJECT_ROOT

    # 1. 添加 Genie-TTS/src 到 sys.path
    candidates = [
        PROJECT_ROOT / "Genie-TTS" / "src",
        PROJECT_ROOT / "src",
        PROJECT_ROOT.parent / "Genie-TTS" / "src",
    ]
    for genie_src in candidates:
        if genie_src.is_dir() and str(genie_src) not in sys.path:
            sys.path.insert(0, str(genie_src))
            logger.info(f"[Genie Backend] 添加 Genie-TTS src 到 sys.path: {genie_src}")
            break

    # 2. 设置 GenieData 目录（避免 Resources.py 的 input() 阻塞）
    genie_data = str(PROJECT_ROOT / "GenieData")
    os.environ.setdefault("GENIE_DATA_DIR", genie_data)

    # 3. 临时 mock input() 防止 Resources.py 阻塞
    original_input = builtins.input
    builtins.input = lambda *a, **kw: "n"
    try:
        import genie_tts  # noqa: F401
        return True
    except ImportError as e:
        logger.error(f"[Genie Backend] 无法导入 genie_tts: {e}")
        return False
    finally:
        builtins.input = original_input


class GeniePipeline(BasePipeline):
    """Genie-TTS ONNX 推理后端

    模型加载策略：
    - init_vits_weights(path) 中 path 指向 ONNX 模型目录
    - 角色名从 VoiceConfig.voice.id 获取
    - 参考音频从 VoiceConfig.ref_audio 获取
    """

    def __init__(self):
        if not _ensure_genie_importable():
            raise ImportError(
                "无法导入 genie_tts 包。请确保 Genie-TTS/src 在 sys.path 中，"
                "或在 settings.toml 中正确配置 project_root。"
            )

        from genie_tts.Core.Inference import tts_client
        from genie_tts.ModelManager import model_manager
        from genie_tts.Audio.ReferenceAudio import ReferenceAudio
        from genie_tts.Utils.TextSplitter import TextSplitter

        self._tts_client = tts_client
        self._model_manager = model_manager
        self._ReferenceAudio = ReferenceAudio
        self._text_splitter = TextSplitter(max_len=40, min_len=5)

        self._current_character: str = ""
        self._current_model_dir: str = ""

        # 默认切分方法（Genie 只用自带的 TextSplitter）
        self._cut_method_names = ["cut0", "cut5"]

        logger.info("[Genie Backend] 初始化完成")

    @property
    def cut_method_names(self) -> List[str]:
        return self._cut_method_names

    def _map_language(self, lang: str) -> str:
        """将 api_neko 语言格式映射为 Genie 格式"""
        return _LANG_MAP.get(lang.lower(), "japanese")

    def _load_character_if_needed(self, character_name: str, model_dir: str, language: str) -> bool:
        """按需加载角色模型"""
        if not model_dir:
            logger.error("[Genie Backend] model_dir 为空，无法加载模型")
            return False

        if self._model_manager.has_character(character_name):
            return True

        logger.info(f"[Genie Backend] 加载角色模型: {character_name} from {model_dir}")
        return self._model_manager.load_character(character_name, model_dir, language)

    def run(self, req: dict) -> Generator[Tuple[int, np.ndarray], None, None]:
        """执行 TTS 推理，按句分段 yield (sample_rate, audio_data)"""
        text = req.get("text", "")
        ref_audio_path = req.get("ref_audio_path", "")
        prompt_text = req.get("prompt_text", "")
        prompt_lang = req.get("prompt_lang", "all_zh")
        text_lang = req.get("text_lang", "all_zh")

        # 语言映射
        genie_text_lang = self._map_language(text_lang)
        genie_prompt_lang = self._map_language(prompt_lang)

        # 获取角色模型
        character_name = self._current_character or "default"
        gsv_model = self._model_manager.get(character_name)
        if gsv_model is None:
            logger.error(f"[Genie Backend] 角色 {character_name} 的模型未加载")
            return

        # 准备参考音频
        try:
            prompt_audio = self._ReferenceAudio(ref_audio_path, prompt_text, genie_prompt_lang)
        except Exception as e:
            logger.error(f"[Genie Backend] 加载参考音频失败: {e}")
            return

        # 按句切分文本
        sentences = self._text_splitter.split(text)
        if not sentences:
            sentences = [text]  # fallback: 不切分

        logger.info(f"[Genie Backend] 文本切分为 {len(sentences)} 句")

        # 逐句推理，每句完成立即 yield
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            try:
                self._tts_client.stop_event.clear()
                audio_chunk = self._tts_client.tts(
                    text=sentence,
                    prompt_audio=prompt_audio,
                    encoder=gsv_model.T2S_ENCODER,
                    first_stage_decoder=gsv_model.T2S_FIRST_STAGE_DECODER,
                    stage_decoder=gsv_model.T2S_STAGE_DECODER,
                    vocoder=gsv_model.VITS,
                    prompt_encoder=gsv_model.PROMPT_ENCODER,
                    language=genie_text_lang,
                )

                if audio_chunk is not None:
                    # Genie 输出为 float32 ndarray，采样率 32000
                    audio_data = audio_chunk.squeeze()
                    if audio_data.dtype != np.int16:
                        # 转换为 int16（与 GSV 后端输出格式一致）
                        audio_data = (audio_data * 32767).astype(np.int16)
                    yield (32000, audio_data)

            except Exception as e:
                logger.error(f"[Genie Backend] 第 {i+1} 句推理失败: {e}", exc_info=True)
                continue

    def init_t2s_weights(self, path: str) -> None:
        """Genie 不分开加载 T2S 权重，忽略"""
        logger.debug(f"[Genie Backend] init_t2s_weights 被调用但 Genie 不需要单独加载 T2S: {path}")

    def init_vits_weights(self, path: str) -> None:
        """加载 ONNX 模型目录

        path 格式：ONNX 模型目录路径（包含 t2s_encoder_fp32.onnx 等文件）
        角色名从目录名推断，或使用 VoiceConfig.voice.id
        """
        if not path:
            return

        model_dir = Path(path)
        if not model_dir.is_dir():
            logger.error(f"[Genie Backend] 模型目录不存在: {path}")
            return

        # 从目录名推断角色名
        character_name = model_dir.name.lower()
        self._current_character = character_name
        self._current_model_dir = str(model_dir)

        # 使用默认语言加载，实际推理时会使用请求中的语言
        success = self._load_character_if_needed(character_name, str(model_dir), "japanese")
        if success:
            logger.info(f"[Genie Backend] 模型加载成功: {character_name} ({path})")
        else:
            logger.error(f"[Genie Backend] 模型加载失败: {path}")

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            name="Genie-TTS (ONNX)",
            version="1.0",
            supported_languages=_SUPPORTED_LANGUAGES,
            supports_streaming=True,
        )

    def validate_params(self, req: dict) -> Optional[str]:
        """校验推理参数"""
        text = req.get("text", "")
        text_lang = req.get("text_lang", "")
        ref_audio_path = req.get("ref_audio_path", "")
        media_type = req.get("media_type", "wav")

        if not ref_audio_path:
            return "ref_audio_path is required"
        if not text:
            return "text is required"
        if not text_lang:
            return "text_lang is required"
        if text_lang.lower() not in _LANG_MAP:
            return f"text_lang: {text_lang} is not supported by Genie backend"
        if media_type not in ("wav", "raw", "ogg", "aac"):
            return f"media_type: {media_type} is not supported"
        return None
