"""Thread execution pool."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, Callable
import signal, psutil

from .base import BasePool, apply_target

__all__ = ('TaskPool',)

if TYPE_CHECKING:
    from typing import TypedDict

    PoolInfo = TypedDict('PoolInfo', {'max-concurrency': int, 'threads': int})

    # `TargetFunction` should be a Protocol that represents fast_trace_task and
    # trace_task_ret.
    TargetFunction = Callable[..., Any]


class ApplyResult:
    def __init__(self, future: Future) -> None:
        self.f = future
        self.get = self.f.result

    def wait(self, timeout: float | None = None) -> None:
        wait([self.f], timeout)


class TaskPool(BasePool):
    """Thread Task Pool."""
    limit: int

    body_can_be_buffer = True
    signal_safe = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=self.limit)

    def on_stop(self) -> None:
        self.executor.shutdown()
        super().on_stop()

    def on_apply(
        self,
        target: TargetFunction,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        callback: Callable[..., Any] | None = None,
        accept_callback: Callable[..., Any] | None = None,
        **_: Any
    ) -> ApplyResult:
        f = self.executor.submit(apply_target, target, args, kwargs,
                                 callback, accept_callback)
        return ApplyResult(f)

    def _get_info(self) -> PoolInfo:
        info = super()._get_info()
        info.update({
            'max-concurrency': self.limit,
            'threads': len(self.executor._threads)
        })
        return info

    def terminate_job(self, pid, signal=None):
        # wait=True, 等待池内所有任务执行完毕回收完资源后才继续
        # wait=False，立即返回，并不会等待池内的任务执行完毕
        # 但无论wait是什么参数，整个程序都会等到所有任务执行完毕
        # self.executor.shutdown(wait=False, cancel_futures=False)
        # self.executor.shutdown(wait=False, cancel_futures=False)
        # self.executor.shutdown会导致无法提交后续任务
        self.kill_child_processes(pid)
        
    def kill_child_processes(self, parent_pid, sig=signal.SIGTERM):
        try:
            parent = psutil.Process(parent_pid)
            print("Will the process will run here.")
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for process in children:
            process.send_signal(sig)
