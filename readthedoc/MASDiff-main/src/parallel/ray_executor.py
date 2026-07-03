from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from src.parallel.base import ParallelExecutor

T = TypeVar("T")
R = TypeVar("R")


def _callable_name(fn: Any) -> str:
    """
    尽量给出一个稳定的“任务名”，用于按任务名匹配 Ray 资源配置。
    - 普通函数：fn.__name__
    - functools.partial：partial.func.__name__
    - 其它：type(fn).__name__
    """
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and name:
        return name
    inner = getattr(fn, "func", None)  # e.g. functools.partial
    inner_name = getattr(inner, "__name__", None)
    if isinstance(inner_name, str) and inner_name:
        return inner_name
    return type(fn).__name__


class RayExecutor(ParallelExecutor):
    """
    基于 Ray 的并行执行器。

    设计目标：
    - runner.py 只依赖 ParallelExecutor.map，因此这里实现 map(fn, items) 语义。
    - 支持按“任务名”（fn.__name__）配置资源，例如：
      - build_initial_individual（构建初始种群个体）
      - mutate_one（精英个体变异）
    - 通过 Ray 的 task options 控制 CPU/GPU 等资源占用。

    配置示例（YAML 中 parallel_executor.kwargs）：
      ray_init_kwargs:
        num_cpus: 16
        num_gpus: 1
        object_store_memory: 8000000000   # bytes
      default_task_options:
        num_cpus: 1
      task_options_by_name:
        build_initial_individual:
          num_cpus: 4
          num_gpus: 1
        mutate_one:
          num_cpus: 4
          num_gpus: 1
    """

    def __init__(
        self,
        *,
        ray_init_kwargs: dict[str, Any] | None = None,
        default_task_options: dict[str, Any] | None = None,
        task_options_by_name: dict[str, dict[str, Any]] | None = None,
        show_progress: bool = True,
        desc: str = "RayExecutor",
        max_updates: int = 20,
        shutdown_on_close: bool = True,
    ) -> None:
        self.show_progress = bool(show_progress)
        self.desc = str(desc)
        self.max_updates = int(max_updates)
        self.shutdown_on_close = bool(shutdown_on_close)

        self._default_task_options = dict(default_task_options or {})
        self._task_options_by_name = {str(k): dict(v) for k, v in (task_options_by_name or {}).items()}

        # 延迟导入：避免用户未安装 ray 时，import 直接失败
        try:
            import ray  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "无法导入 ray。请先安装 Ray（例如 `pip install ray[default]`），"
                "或在配置中继续使用 SerialExecutor。"
            ) from e

        self._ray = ray
        self._owns_ray = False

        if not ray.is_initialized():
            ray.init(**(ray_init_kwargs or {}))
            self._owns_ray = True

    def _options_for(self, fn: Callable[[T], R]) -> dict[str, Any]:
        name = _callable_name(fn)
        # 任务级 options：默认 + 按名字覆盖（同名 key 覆盖默认）
        opts = dict(self._default_task_options)
        by_name = self._task_options_by_name.get(name)
        if by_name:
            opts.update(by_name)
        return opts

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        ray = self._ray
        seq = list(items)
        total = len(seq)
        if total == 0:
            return []

        options = self._options_for(fn)

        remote_fn = ray.remote(fn).options(**options) if options else ray.remote(fn)
        refs = [remote_fn.remote(x) for x in seq]

        if self.show_progress:
            max_updates = max(1, self.max_updates)
            step = max(1, total // max_updates)
            print(f"{self.desc}({ _callable_name(fn) }): 0/{total}")

            pending = list(refs)
            done = 0
            # 用 ray.wait 做增量进度；结果最终按 refs 顺序 ray.get
            while pending:
                ready, pending = ray.wait(pending, num_returns=1, timeout=None)
                done += len(ready)
                if done == total or (done % step == 0):
                    print(f"{self.desc}({ _callable_name(fn) }): {done}/{total}")

        # 保序返回
        return list(ray.get(refs))

    def close(self) -> None:
        if not self.shutdown_on_close:
            return None
        # 仅在由本执行器初始化 Ray 时 shutdown，避免影响外部 Ray 会话
        if not getattr(self, "_owns_ray", False):
            return None
        try:
            self._ray.shutdown()
        except Exception:
            return None
        return None

