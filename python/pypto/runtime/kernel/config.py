# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Immutable configuration for the public borrowed-context Worker."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelConfig:
    """Immutable process configuration; no callable or per-call stream is stored."""

    platform: str
    runtime: str
    device_id: int
    aicpu_thread_num: int = 0
    enable_chip_swimlane: int | bool = 0
    enable_dep_gen: bool = False
    output_dir: str | Path | None = None
    # Ring arena sizing is fixed at initialization and reused by later calls.
    ring_task_window: int | Sequence[int] | None = None
    ring_heap: int | Sequence[int] | None = None
    ring_dep_pool: int | Sequence[int] | None = None

    workspace_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        from pypto.runtime.runner import _normalize_swimlane_level  # noqa: PLC0415

        if self.platform != "a2a3" or self.runtime not in ("tensormap_and_ringbuffer", "host_build_graph"):
            raise ValueError(
                f"Kernel initialization supports a2a3 TMR/HBG, got {self.platform}/{self.runtime}"
            )
        object.__setattr__(
            self,
            "enable_chip_swimlane",
            _normalize_swimlane_level(self.enable_chip_swimlane, "enable_chip_swimlane"),
        )
        if type(self.enable_dep_gen) is not bool:
            raise TypeError(f"enable_dep_gen must be bool, got {self.enable_dep_gen!r}")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        if (self.enable_chip_swimlane or self.enable_dep_gen) and self.output_dir is None:
            raise ValueError("Kernel DFX requires output_dir when swimlane or dep_gen is enabled")
        if type(self.device_id) is not int or self.device_id < 0:
            raise ValueError(f"Expected a nonnegative kernel device id, got {self.device_id!r}")
        if type(self.aicpu_thread_num) is not int or self.aicpu_thread_num not in (0, 2, 3, 4, 5):
            raise ValueError(f"Expected auto (0) or 2..5 AICPU threads, got {self.aicpu_thread_num!r}")
        if self.workspace_budget_bytes is not None and (
            type(self.workspace_budget_bytes) is not int or self.workspace_budget_bytes < 1024
        ):
            raise ValueError("workspace_budget_bytes must be an integer of at least 1024 bytes")
        for field in ("ring_task_window", "ring_heap", "ring_dep_pool"):
            value = getattr(self, field)
            if value is None:
                continue
            if type(value) is int:
                value = (value,) * 4
            else:
                value = tuple(value)
                if len(value) != 4:
                    raise ValueError(
                        f"{field} must be a scalar or exactly 4 per-ring values, got {len(value)}"
                    )
            if any(type(v) is not int or v <= 0 for v in value):
                raise ValueError(f"{field} entries must be positive ints, got {value}")
            object.__setattr__(self, field, value)
