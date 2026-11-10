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

from api_neko.audio_utils import pack_audio, WavStreamEncoder


class TaskStatus(str, Enum):
    QUEUED = "queued"
    LOADING_MODEL = "loading_model"
    INFERRING = "inferring"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


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

    async def wait_until_done(self, timeout: Optional[float] = 120.0) -> bool:
        """Measure a real deadline, regardless of how often audio wakes us."""
        async def wait():
            while not self.done:
                await self.wait_for_update()
        try:
            await asyncio.wait_for(wait(), timeout=timeout)
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
    pcm48_wav: bool = False
    retain_result: bool = True
    model_only: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    release_when_done: bool = False
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

    def __init__(self, tts_pipeline, validator, cut_method_names: list, *, max_completed_tasks: int = 100):
        self._pipeline = tts_pipeline
        self._validator = validator  # 实现了 validate_params(req) 的对象
        self._cut_method_names = cut_method_names

        self._queue: asyncio.Queue[InferTask] = asyncio.Queue()
        self._tasks: dict[int, InferTask] = {}
        self._task_counter = 0
        self._counter_lock = threading.Lock()

        self._manual_weights = tts_pipeline.get_model_paths()
        self._current_gpt, self._current_sovits = self._manual_weights

        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._active_task: Optional[InferTask] = None
        self._max_completed_tasks = max_completed_tasks

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
        for task_id in list(self._tasks):
            self.cancel_task(task_id)
        if self._worker_task:
            # Cancelling to_thread's waiter would leave GPU work running. Let
            # the cooperative stop finish before retiring the shared pipeline.
            await asyncio.shield(self._worker_task)
        while not self._queue.empty():
            task = self._queue.get_nowait()
            task.status = TaskStatus.CANCELLED
            task.output.mark_error("Task cancelled")
            self._tasks.pop(task.task_id, None)
            self._queue.task_done()

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
        *,
        pcm48_wav: bool = False,
        retain_result: bool = True,
        model_only: bool = False,
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
            pcm48_wav=pcm48_wav,
            retain_result=retain_result,
            model_only=model_only,
        )
        self._tasks[task_id] = task
        self._prune_completed_tasks()
        await self._queue.put(task)
        return task

    async def submit_selected(self, tts_params: dict, media_type: str = "wav") -> InferTask:
        """Snapshot the manually selected pair so v3 requests cannot replace it."""
        gpt, sovits = self._manual_weights
        return await self.submit("_manual", gpt, sovits, tts_params,
                                 media_type=media_type, retain_result=False)

    async def select_model_weights(self, *, gpt: str = "", sovits: str = ""):
        """Serialize manual model selection with every other pipeline operation."""
        task = await self.submit("_manual", gpt, sovits, {}, model_only=True, retain_result=False)
        try:
            await task.output.wait_until_done(timeout=None)
            if task.output.error:
                raise ValueError(task.output.error)
        finally:
            self.cancel_task(task.task_id)
            self.cleanup_task(task.task_id)

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
        task = self._tasks.get(task_id)
        if task is not None:
            if task.output.done:
                self._tasks.pop(task_id)
            else:
                task.release_when_done = True

    def cancel_task(self, task_id: int):
        """Cancel only this task; idle/probe connections cannot stop other clients."""
        task = self._tasks.get(task_id)
        if task is None or task.output.done:
            return
        task.cancelled.set()
        if self._active_task is task and task.status == TaskStatus.INFERRING:
            self._pipeline.stop()

    def _prune_completed_tasks(self):
        completed = [task_id for task_id, task in self._tasks.items()
                     if task.retain_result and task.output.done]
        for task_id in completed[:max(0, len(completed) - self._max_completed_tasks)]:
            self._tasks.pop(task_id)

    def _publish_chunk(self, task: InferTask, chunk: AudioChunk):
        if not task.cancelled.is_set():
            task.output.add_chunk(chunk)

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
                self._active_task = task
                if not task.cancelled.is_set():
                    await self._process_task(task)
                task.status = TaskStatus.CANCELLED if task.cancelled.is_set() else TaskStatus.DONE
            except Exception as e:
                task.output.error = f"Worker error: {e}"
                task.status = TaskStatus.ERROR
            finally:
                if task.cancelled.is_set():
                    task.status = TaskStatus.CANCELLED
                    task.output.error = "Task cancelled"
                task.output.mark_done()
                self._active_task = None
                if task.release_when_done:
                    self._tasks.pop(task.task_id, None)
                self._prune_completed_tasks()
                self._queue.task_done()

    async def _process_task(self, task: InferTask):
        """处理单个推理任务"""
        # 1. 模型切换（如果需要）
        gpt, sovits = task.gpt_weights, task.sovits_weights
        if task.model_only:
            gpt = gpt or self._manual_weights[0]
            sovits = sovits or self._manual_weights[1]
        need_gpt = gpt and gpt != self._current_gpt
        need_sovits = sovits and sovits != self._current_sovits

        if need_gpt or need_sovits:
            task.status = TaskStatus.LOADING_MODEL
            if need_sovits:
                self._current_sovits = ""
                await asyncio.to_thread(self._pipeline.init_vits_weights, sovits)
                self._current_sovits = sovits
            if task.cancelled.is_set():
                return
            if need_gpt:
                self._current_gpt = ""
                await asyncio.to_thread(self._pipeline.init_t2s_weights, gpt)
                self._current_gpt = gpt

        if task.cancelled.is_set():
            return
        if task.model_only:
            self._manual_weights = gpt, sovits
            return
        error = self.validate_params({**task.tts_params, "media_type": task.media_type})
        if error:
            raise ValueError(error)

        # 2. 推理
        task.status = TaskStatus.INFERRING

        req = dict(task.tts_params)

        media_type = task.media_type
        chunk_index = 0

        def _run_inference():
            """在线程中执行推理（阻塞操作）"""
            nonlocal chunk_index
            if task.cancelled.is_set():
                return
            encoder = WavStreamEncoder() if task.pcm48_wav else None
            tts_generator = self._pipeline.run(req)

            def publish(data, rate):
                nonlocal chunk_index
                chunk = AudioChunk(index=chunk_index, data=data, sample_rate=rate)
                chunk_index += 1
                self._loop.call_soon_threadsafe(self._publish_chunk, task, chunk)

            try:
                for sr, audio_data in tts_generator:
                    if task.cancelled.is_set():
                        break
                    if encoder is not None:
                        for data in encoder.feed(audio_data, sr):
                            publish(data, encoder.sample_rate)
                    else:
                        publish(pack_audio(BytesIO(), audio_data, sr, media_type).getvalue(), sr)
                if encoder is not None and not task.cancelled.is_set():
                    for data in encoder.flush():
                        publish(data, encoder.sample_rate)
                if chunk_index == 0 and not task.cancelled.is_set():
                    raise ValueError("No audio returned by the synthesis backend")
            finally:
                tts_generator.close()

        try:
            await asyncio.to_thread(_run_inference)
        except Exception:
            # The host may reset/reload its models while handling an inference
            # exception. Do not reuse identities that no longer describe it.
            self._current_gpt = self._current_sovits = ""
            raise
