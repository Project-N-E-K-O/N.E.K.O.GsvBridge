"""
GPT-SoVITS API v3 - 音频打包工具函数

从 api_v2.py 提取，供 TTS 路由复用。
支持 wav / ogg / aac / raw 格式的音频编码。
"""

import subprocess
import wave
from io import BytesIO

import numpy as np
import soxr


class WavStreamEncoder:
    """Encode mono PCM as bounded, independently decodable 48 kHz WAV frames."""

    sample_rate = 48000
    frame_samples = 48000

    def __init__(self):
        self._source_rate = None
        self._resampler = None

    def _pack(self, audio):
        pcm = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
        return [pack_wav(BytesIO(), pcm[start:start + self.frame_samples], self.sample_rate).getvalue()
                for start in range(0, len(pcm), self.frame_samples)]

    def feed(self, audio: np.ndarray, source_rate: int) -> list[bytes]:
        if self._source_rate is None:
            self._source_rate = source_rate
            if source_rate != self.sample_rate:
                self._resampler = soxr.ResampleStream(source_rate, self.sample_rate, 1, dtype="float32")
        elif source_rate != self._source_rate:
            raise ValueError("Sample rate changed inside one synthesis task")
        audio = np.asarray(audio)
        if audio.ndim != 1:
            raise ValueError("TTS output must be mono")
        if audio.dtype == np.int16:
            # Preserve int16 exactly for native 48 kHz output.
            if self._resampler is None:
                return [pack_wav(BytesIO(), audio[start:start + self.frame_samples], self.sample_rate).getvalue()
                        for start in range(0, len(audio), self.frame_samples)]
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)
        if self._resampler is not None:
            audio = self._resampler.resample_chunk(audio)
        return self._pack(audio)

    def flush(self) -> list[bytes]:
        if self._resampler is None:
            return []
        tail = self._resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
        self._resampler = None
        return self._pack(tail)


def join_wav_chunks(chunks: list[bytes]) -> bytes:
    """Return one WAV file rather than concatenating multiple RIFF headers."""
    output = BytesIO()
    with wave.open(output, "wb") as writer:
        params = None
        for chunk in chunks:
            with wave.open(BytesIO(chunk), "rb") as reader:
                current = (reader.getnchannels(), reader.getsampwidth(), reader.getframerate())
                if params is None:
                    params = current
                    writer.setnchannels(params[0])
                    writer.setsampwidth(params[1])
                    writer.setframerate(params[2])
                elif current != params:
                    raise ValueError("Inconsistent WAV formats in one task")
                writer.writeframes(reader.readframes(reader.getnframes()))
    return output.getvalue()


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    """将音频数据编码为 OGG 格式（通过 ffmpeg 转码，无需 libsndfile）"""
    # 确保数据为 int16
    if data.dtype != np.int16:
        data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f", "s16le",
            "-ar", str(rate),
            "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libvorbis",
            "-q:a", "4",
            "-f", "ogg",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    """将音频数据写入为原始 PCM 字节"""
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    """将音频数据编码为 WAV 格式（使用标准库 wave）"""
    # 确保数据为 int16
    if data.dtype != np.int16:
        data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)

    io_buffer = BytesIO()
    with wave.open(io_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
    """将音频数据编码为 AAC 格式（需要 ffmpeg）"""
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f", "s16le",
            "-ar", str(rate),
            "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "aac",
            "-b:a", "192k",
            "-vn",
            "-f", "adts",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str) -> BytesIO:
    """根据 media_type 选择对应的编码方式"""
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def wave_header_chunk(frame_input: bytes = b"", channels: int = 1, sample_width: int = 2, sample_rate: int = 32000) -> bytes:
    """生成 WAV 文件头（用于流式 WAV 输出的首个 chunk）"""
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()
