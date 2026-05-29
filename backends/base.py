"""
TTS 推理后端抽象基类

所有后端（GSV PyTorch、Genie ONNX 等）都需要实现此接口，
供 InferenceEngine 统一调用。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Tuple, Optional, List

import numpy as np


@dataclass
class BackendInfo:
    """后端基本信息"""
    name: str
    version: str
    supported_languages: List[str]
    supports_streaming: bool = False


class BasePipeline(ABC):
    """TTS 推理后端抽象基类

    InferenceEngine 通过以下接口与后端交互：
    - run(req)：执行推理，yield (sample_rate, audio_ndarray)
    - init_t2s_weights(path)：加载 T2S (GPT) 模型权重
    - init_vits_weights(path)：加载 VITS (SoVITS) 模型权重
    - get_info()：获取后端信息
    - validate_params(req)：校验推理参数
    """

    @abstractmethod
    def run(self, req: dict) -> Generator[Tuple[int, np.ndarray], None, None]:
        """执行 TTS 推理

        Args:
            req: 推理参数 dict，由 VoiceConfig.to_tts_inputs() 生成

        Yields:
            (sample_rate, audio_data) 元组，audio_data 为 np.ndarray
        """
        ...

    @abstractmethod
    def init_t2s_weights(self, path: str) -> None:
        """加载 T2S (GPT) 模型权重

        Args:
            path: 模型权重文件路径
        """
        ...

    @abstractmethod
    def init_vits_weights(self, path: str) -> None:
        """加载 VITS (SoVITS) 模型权重

        Args:
            path: 模型权重文件路径
        """
        ...

    @abstractmethod
    def get_info(self) -> BackendInfo:
        """获取后端信息"""
        ...

    def validate_params(self, req: dict) -> Optional[str]:
        """校验推理参数，返回错误信息或 None

        子类可覆盖此方法以添加后端特定的校验逻辑。
        默认实现只检查必填字段。
        """
        if not req.get("text"):
            return "text is required"
        if not req.get("ref_audio_path"):
            return "ref_audio_path is required"
        return None
