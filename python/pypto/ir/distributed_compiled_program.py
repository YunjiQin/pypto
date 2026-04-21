# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Distributed compiled program wrapper for L3+ programs.

Provides a callable API similar to :class:`CompiledProgram` but executes
through simpler's distributed runtime (Worker level=3)::

    compiled = ir.compile(MyDistributedProgram)
    compiled(a, b, c)   # executes via simpler Worker(level=3)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pypto.backend import BackendType
from pypto.pypto_core.ir import Program, Role, level_to_linqu_level

from .compiled_program import CallArg, _default_platform, _extract_param_infos, _ParamInfo, _to_torch_dtype


def _extract_param_infos_from_func(func):
    """Extract parameter metadata from a specific function."""
    from pypto.pypto_core.ir import ConstInt, ParamDirection, ScalarType, ShapedType  # noqa: PLC0415

    param_infos = []
    output_indices = []

    for i, (param, direction) in enumerate(zip(func.params, func.param_directions, strict=True)):
        param_type = param.type
        shape = None

        if isinstance(param_type, ShapedType):
            dtype = param_type.dtype
            shape = [dim.value if isinstance(dim, ConstInt) else -1 for dim in param_type.shape]
        elif isinstance(param_type, ScalarType):
            dtype = param_type.dtype
        else:
            raise TypeError(
                f"Unsupported parameter type for {param.name_hint!r}: {type(param_type).__name__}"
            )

        param_infos.append(_ParamInfo(name=param.name_hint, direction=direction, shape=shape, dtype=dtype))
        if direction == ParamDirection.Out:
            output_indices.append(i)

    return param_infos, output_indices, list(func.return_types)


@dataclass
class DistributedConfig:
    """Configuration for L3 distributed execution."""

    device_ids: list[int] = field(default_factory=lambda: [0])
    num_sub_workers: int = 0
    runtime: str = "tensormap_and_ringbuffer"
    block_dim: int = 1
    aicpu_thread_num: int = 1


class DistributedCompiledProgram:
    """A compiled L3+ distributed program that executes via simpler Worker(level=3).

    Returned by :func:`ir.compile` when the program contains HOST-level
    or higher hierarchy functions.

    Calling conventions match :class:`CompiledProgram`:

    **In-place** (output passed as argument)::

        compiled(a, b, c)

    **Return** (program has a return value)::

        c = compiled(a, b)
    """

    __test__ = False

    def __init__(
        self,
        program: Program,
        output_dir: str,
        *,
        backend_type: BackendType = BackendType.Ascend910B,
        platform: str | None = None,
        distributed_config: DistributedConfig | None = None,
        transformed_program: Program | None = None,
    ) -> None:
        self._program = program
        self._transformed_program = transformed_program or program
        self._output_dir = Path(output_dir).resolve()
        self._backend_type = backend_type
        self._platform = platform or _default_platform(backend_type)
        self._distributed_config = distributed_config or DistributedConfig()
        self._param_infos = None
        self._output_indices = None
        self._return_types = None

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def program(self) -> Program:
        return self._program

    @property
    def platform(self) -> str:
        return self._platform

    def __str__(self) -> str:
        return str(self._output_dir)

    def __repr__(self) -> str:
        return f"DistributedCompiledProgram({self._output_dir!s})"

    def __fspath__(self) -> str:
        return str(self._output_dir)

    def _get_metadata(self) -> tuple[list[_ParamInfo], list[int], list[Any]]:
        if self._param_infos is None:
            # Find the HOST orchestrator function in the transformed program
            # (uses post-SSA names that match the generated Python code)
            host_orch = None
            for func in self._transformed_program.functions.values():
                if (
                    func.level is not None
                    and level_to_linqu_level(func.level) >= 3
                    and func.role is not None
                    and func.role == Role.Orchestrator
                ):
                    host_orch = func
                    break

            if host_orch is not None:
                self._param_infos, self._output_indices, self._return_types = _extract_param_infos_from_func(
                    host_orch
                )
            else:
                self._param_infos, self._output_indices, self._return_types = _extract_param_infos(
                    self._transformed_program
                )
        assert self._output_indices is not None and self._return_types is not None
        return self._param_infos, self._output_indices, self._return_types

    def __call__(
        self,
        *args: CallArg,
        config: Any = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Execute the distributed program via simpler Worker(level=3)."""
        from pypto.runtime.distributed_runner import execute_distributed  # noqa: PLC0415

        param_infos, output_indices, return_types = self._get_metadata()
        n_params = len(param_infos)
        n_inputs = n_params - len(output_indices)
        has_return = len(return_types) > 0
        return_style = has_return and len(args) == n_inputs

        if len(args) == n_params:
            all_args: list[CallArg] = list(args)
        elif return_style:
            all_args = self._build_full_args(args, param_infos, output_indices)
        else:
            expected = f"{n_params} (in-place)"
            if has_return:
                expected += f" or {n_inputs} (return)"
            raise TypeError(
                f"DistributedCompiledProgram expects {expected} arguments, got {len(args)}. "
                f"Parameters: {[p.name for p in param_infos]}"
            )

        # Validate and coerce args
        coerced: list[torch.Tensor] = []
        for info, arg in zip(param_infos, all_args, strict=True):
            if not isinstance(arg, torch.Tensor):
                raise TypeError(
                    f"Distributed programs only support tensor parameters. "
                    f"Parameter {info.name!r} got {type(arg).__name__}"
                )
            coerced.append(arg)

        execute_distributed(self, coerced, config)

        if not return_style:
            return None
        outputs = [coerced[i] for i in output_indices]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    @staticmethod
    def _build_full_args(input_args, param_infos, output_indices):
        output_set = set(output_indices)
        all_tensors = []
        input_idx = 0

        for i, info in enumerate(param_infos):
            if i in output_set:
                if info.shape is None:
                    raise ValueError(f"Cannot allocate output tensor {info.name!r}: no shape in IR")
                if any(d < 0 for d in info.shape):
                    raise ValueError(
                        f"Cannot allocate output tensor {info.name!r}: shape {info.shape} "
                        f"contains dynamic dimensions."
                    )
                torch_dtype = _to_torch_dtype(info.dtype)
                if torch_dtype is None:
                    raise ValueError(f"Unsupported dtype {info.dtype} for output tensor {info.name!r}")
                all_tensors.append(torch.zeros(info.shape, dtype=torch_dtype))
            else:
                all_tensors.append(input_args[input_idx])
                input_idx += 1

        return all_tensors
