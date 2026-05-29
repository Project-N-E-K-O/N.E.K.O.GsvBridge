"""
GPT-SoVITS PyTorch 推理后端

包装原项目的 TTS pipeline，实现 BasePipeline 接口。
这是默认后端，依赖 GPT_SoVITS 包。
"""

import logging
from typing import Generator, Tuple, Optional, List

import numpy as np

from .base import BasePipeline, BackendInfo

logger = logging.getLogger(__name__)


class GSVPipeline(BasePipeline):
    """GPT-SoVITS 原项目 PyTorch 推理后端"""

    def __init__(self, config_path: str):
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        self._tts_config = TTS_Config(config_path)
        logger.info(f"[GSV Backend] 加载配置: {config_path}")
        logger.info("\n"+str(self._tts_config))

        self._tts = TTS(self._tts_config)
        self._cut_method_names = self._load_cut_method_names()

    @staticmethod
    def _load_cut_method_names() -> List[str]:
        """加载可用的文本切分方法名称"""
        try:
            from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names
            return get_method_names()
        except ImportError:
            logger.warning("[GSV Backend] 无法加载切分方法，使用默认列表")
            return ["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]

    @property
    def tts_config(self):
        """暴露 TTS_Config 供外部使用（如 InferenceEngine 校验参数）"""
        return self._tts_config

    @property
    def cut_method_names(self) -> List[str]:
        return self._cut_method_names

    def run(self, req: dict) -> Generator[Tuple[int, np.ndarray], None, None]:
        yield from self._tts.run(req)

    def init_t2s_weights(self, path: str) -> None:
        logger.info(f"[GSV Backend] 加载 GPT 权重: {path}")
        self._tts.init_t2s_weights(path)

    def init_vits_weights(self, path: str) -> None:
        logger.info(f"[GSV Backend] 加载 SoVITS 权重: {path}")
        self._tts.init_vits_weights(path)

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            name="GPT-SoVITS (PyTorch)",
            version=getattr(self._tts_config, 'version', 'unknown'),
            supported_languages=getattr(self._tts_config, 'languages', []),
            supports_streaming=True,
        )

    def validate_params(self, req: dict) -> Optional[str]:
        """使用原项目的校验逻辑"""
        text = req.get("text", "")
        text_lang = req.get("text_lang", "")
        ref_audio_path = req.get("ref_audio_path", "")
        prompt_lang = req.get("prompt_lang", "")
        media_type = req.get("media_type", "wav")
        text_split_method = req.get("text_split_method", "cut5")

        if not ref_audio_path:
            return "ref_audio_path is required"
        if not text:
            return "text is required"
        if not text_lang:
            return "text_lang is required"
        if text_lang.lower() not in self._tts_config.languages:
            return f"text_lang: {text_lang} is not supported in version {self._tts_config.version}"
        if not prompt_lang:
            return "prompt_lang is required"
        if prompt_lang.lower() not in self._tts_config.languages:
            return f"prompt_lang: {prompt_lang} is not supported in version {self._tts_config.version}"
        if media_type not in ("wav", "raw", "ogg", "aac"):
            return f"media_type: {media_type} is not supported"
        if text_split_method not in self._cut_method_names:
            return f"text_split_method: {text_split_method} is not supported"
        return None
