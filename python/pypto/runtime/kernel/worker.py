# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Thin public Worker adapter; Simpler owns callable slots and native run lifetime."""

from typing import Any

from ._compat import require_runtime
from .config import KernelConfig
from .owner import _OwnerThread


def make_call_config(config: KernelConfig, factory: Any) -> Any:
    """Validate the shared runtime config before native initialization claims a mode."""
    cfg = factory()
    cfg.aicpu_thread_num = config.aicpu_thread_num
    cfg.enable_chip_swimlane = config.enable_chip_swimlane
    cfg.enable_dep_gen = config.enable_dep_gen
    for field in ("ring_task_window", "ring_heap", "ring_dep_pool"):
        value = getattr(config, field)
        if value is not None:
            setattr(cfg.runtime_env, field, list(value))
    if config.output_dir is not None:
        cfg.output_prefix = str(config.output_dir)
    cfg.validate()
    return cfg


class _NativeWorker:
    def __init__(self, config: KernelConfig, state: Any) -> None:
        from pypto.torch.shutdown import install_shutdown  # noqa: PLC0415

        self._api = require_runtime()
        self._native = install_shutdown(state)
        self._context = self._native.borrow_context(config.device_id)
        self._device_id = config.device_id
        self._generation = state.generation
        self.worker: Any = None
        self._owner = _OwnerThread()

    def check_context(self) -> None:
        if self._native.borrow_context(self._device_id) != self._context:
            raise RuntimeError("Kernel Worker belongs to a different borrowed context; use a fresh process")

    def on_owner_thread(self) -> bool:
        import threading  # noqa: PLC0415

        return threading.current_thread() is self._owner.thread

    def init(self, config: KernelConfig) -> None:
        self._owner.call(lambda: self._init(config))

    def _init(self, config: KernelConfig) -> None:
        self._native.bind_context(self._context)
        cfg = make_call_config(config, self._api.CallConfig)
        options = {}
        if config.workspace_budget_bytes is not None:
            options["workspace_budget_bytes"] = config.workspace_budget_bytes
        self.worker = self._api.Worker(
            level=2,
            platform=config.platform,
            runtime=config.runtime,
            device_id=config.device_id,
            enable_sdma=False,
            collect_across_runs=bool(config.enable_chip_swimlane or config.enable_dep_gen),
            **options,
        )
        self.worker.init(prewarm_config=cfg, borrowed_context_generation=self._generation)

    def prepare(self, callable_: Any) -> Any:
        def register() -> Any:
            self._native.bind_context(self._context)
            return self.worker.register(callable_)

        return self._owner.call(register)

    @property
    def device_location(self) -> Any:
        return self.worker.device_location

    def wait(self, ticket: Any) -> None:
        def retire() -> None:
            self._native.bind_context(self._context)
            ticket.wait()

        self._owner.call(retire)

    def close(self) -> None:
        def finalize() -> None:
            self._native.bind_context(self._context)
            if self.worker is not None:
                self.worker.close()

        self._owner.call(finalize)
        self._owner.stop()


__all__ = ["_NativeWorker"]
