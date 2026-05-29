"""
GPT-SoVITS API v3 - 推理引擎

双队列架构：Input Queue → Worker → Output Buffer (per task)
- 单 Worker 串行推理，避免 GPU 并发冲突
- 每个 task 有独立的 OutputBuffer，支持流式和一次性获取
- Worker 在推理前自动检测并切换模型
- 自增 task_id + 时间戳，保证有序
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Optional

from api_neko.audio_utils import pack_audio


class TaskStatus(str, Enum):
    QUEUED = "queued"
    LOADING_MODEL = "loading_model"
    INFERRING = "inferring"
    DONE = "done"
    ERROR = "error"


@dataclass
class AudioChunk:
    """单个音频片段"""
    index: int
    data: bytes
    sample_rate: int


@dataclass
class OutputBuffer:
    """每个 task 的音频输出缓冲区"""
    chunks: list[AudioChunk] = field(default_factory=list)
    done: bool = False
    error: Optional[str] = None
    # asyncio.Event 用于通知有新 chunk 到达或任务完成
    # 注意：必须在事件循环中创建
    _notify: Optional[asyncio.Event] = field(default=None, repr=False)

    def __post_init__(self):
        try:
            self._notify = asyncio.Event()
        except RuntimeError:
            # 如果不在事件循环中（不应发生），延迟创建
            self._notify = None

    def set_notify(self, event: asyncio.Event):
        self._notify = event

    def add_chunk(self, chunk: AudioChunk):
        self.chunks.append(chunk)
        if self._notify:
            self._notify.set()

    def mark_done(self):
        self.done = True
        if self._notify:
            self._notify.set()

    def mark_error(self, msg: str):
        self.error = msg
        self.done = True
        if self._notify:
            self._notify.set()

    async def wait_for_update(self, timeout: float = 30.0) -> bool:
        """等待新 chunk 或完成信号，返回是否有更新"""
        if self._notify is None:
            return False
        try:
            await asyncio.wait_for(self._notify.wait(), timeout=timeout)
            self._notify.clear()
            return True
        except asyncio.TimeoutError:
            return False


@dataclass
class InferTask:
    """推理任务"""
    task_id: int
    timestamp: float
    # 声音配置相关
    voice_id: str
    gpt_weights: str
    sovits_weights: str
    # TTS 推理参数（传给 tts_pipeline.run() 的 dict）
    tts_params: dict
    # 输出配置
    media_type: str = "wav"
    # 输出缓冲区（自动创建）
    output: OutputBuffer = field(init=False)
    status: TaskStatus = TaskStatus.QUEUED

    def __post_init__(self):
        self.output = OutputBuffer()


class InferenceEngine:
    """
    推理引擎：管理任务队列和 Worker。

    使用方式：
        engine = InferenceEngine(pipeline, pipeline, cut_method_names)
        await engine.start()
        task = await engine.submit(voice_id, gpt_weights, sovits_weights, tts_params, media_type)
        # 获取结果...
        await engine.stop()
    """

    def __init__(self, tts_pipeline, validator, cut_method_names: list):
        self._pipeline = tts_pipeline
        self._validator = validator  # 实现了 validate_params(req) 的对象
        self._cut_method_names = cut_method_names

        self._queue: asyncio.Queue[InferTask] = asyncio.Queue()
        self._tasks: dict[int, InferTask] = {}
        self._task_counter = 0
        self._counter_lock = threading.Lock()

        self._current_gpt: str = ""
        self._current_sovits: str = ""

        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _next_id(self) -> int:
        with self._counter_lock:
            self._task_counter += 1
            return self._task_counter

    async def start(self):
        """启动 Worker"""
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        """停止 Worker"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def validate_params(self, req: dict) -> Optional[str]:
        """校验 TTS 请求参数，委托给 pipeline 后端的 validate_params"""
        return self._validator.validate_params(req)

    async def submit(
        self,
        voice_id: str,
        gpt_weights: str,
        sovits_weights: str,
        tts_params: dict,
        media_type: str = "wav",
    ) -> InferTask:
        """提交推理任务，返回 InferTask（含 task_id 和 output buffer）"""
        task_id = self._next_id()
        task = InferTask(
            task_id=task_id,
            timestamp=time.time(),
            voice_id=voice_id,
            gpt_weights=gpt_weights,
            sovits_weights=sovits_weights,
            tts_params=tts_params,
            media_type=media_type,
        )
        self._tasks[task_id] = task
        await self._queue.put(task)
        return task

    def get_task(self, task_id: int) -> Optional[InferTask]:
        """获取任务"""
        return self._tasks.get(task_id)

    def get_queue_info(self) -> dict:
        """获取队列状态"""
        return {
            "queue_size": self._queue.qsize(),
            "total_tasks": len(self._tasks),
            "current_gpt": self._current_gpt,
            "current_sovits": self._current_sovits,
        }

    def cleanup_task(self, task_id: int):
        """清理已完成的任务（释放内存）"""
        self._tasks.pop(task_id, None)

    # ─── Worker ───

    async def _worker_loop(self):
        """Worker 主循环：串行消费队列中的任务"""
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._process_task(task)
            except Exception as e:
                task.output.mark_error(f"Worker error: {e}")
                task.status = TaskStatus.ERROR
            finally:
                self._queue.task_done()

    async def _process_task(self, task: InferTask):
        """处理单个推理任务"""
        # 1. 模型切换（如果需要）
        need_gpt = task.gpt_weights and task.gpt_weights != self._current_gpt
        need_sovits = task.sovits_weights and task.sovits_weights != self._current_sovits

        if need_gpt or need_sovits:
            task.status = TaskStatus.LOADING_MODEL
            if need_gpt:
                await asyncio.to_thread(self._pipeline.init_t2s_weights, task.gpt_weights)
                self._current_gpt = task.gpt_weights
            if need_sovits:
                await asyncio.to_thread(self._pipeline.init_vits_weights, task.sovits_weights)
                self._current_sovits = task.sovits_weights

        # 2. 推理
        task.status = TaskStatus.INFERRING

        # 构建推理参数：尊重配置中的 return_fragment / streaming_mode
        # 不强制覆盖，让 TTS.run() 走最优路径（批量推理+并行合成）
        req = dict(task.tts_params)
        req["streaming_mode"] = False

        media_type = task.media_type
        chunk_index = 0

        def _run_inference():
            """在线程中执行推理（阻塞操作）"""
            nonlocal chunk_index
            try:
                tts_generator = self._pipeline.run(req)
                for sr, audio_data in tts_generator:
                    # 每个 chunk 打包为完整 WAV（含 header），
                    # 这样前端 decodeAudioData 可以独立解码每个 chunk
                    buf = BytesIO()
                    data = pack_audio(buf, audio_data, sr, media_type).getvalue()

                    chunk = AudioChunk(index=chunk_index, data=data, sample_rate=sr)
                    chunk_index += 1
                    # 线程安全地通知事件循环
                    if self._loop and self._loop.is_running():
                        self._loop.call_soon_threadsafe(task.output.add_chunk, chunk)
                    else:
                        task.output.add_chunk(chunk)

                # 推理完成
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(task.output.mark_done)
                else:
                    task.output.mark_done()
            except Exception as e:
                msg = str(e)
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(task.output.mark_error, msg)
                else:
                    task.output.mark_error(msg)

        await asyncio.to_thread(_run_inference)
        task.status = TaskStatus.DONE if not task.output.error else TaskStatus.ERROR
