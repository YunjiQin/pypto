# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel call constraints over the shared program parameter signature.

The L2 wire format is Simpler's ChipStorageTaskArgs in both execution modes.
This module owns no native dtype tags or scalar codec. Its descriptor is a
snapshot for artifact validation, not a second parameter type system.
"""

import re
from collections.abc import Sequence
from typing import Any, TypeVar, cast

from pypto._identity import encode_record
from pypto.ir.param_info import (
    ParamInfo,
    _param_info_from_dict,
    _param_info_to_dict,
    _to_torch_dtype,
    bind_complete_args,
)
from pypto.pypto_core.ir import ParamDirection

# Shared L2 limits, checked against the pinned SDK by the host-only ABI UT.
MAX_KERNEL_TENSORS = 256
MAX_KERNEL_SCALARS = 128
MAX_KERNEL_RANK = 5
# These are supported orchestration scalar types, not a bit-encoding table.
_KERNEL_SCALARS = frozenset(
    {"fp32", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "bool", "index"}
)
_Arg = TypeVar("_Arg")


class KernelABI:
    """Snapshot a common signature and its kernel-specific return contract.

    SDK identity must come from the compilation context. It is never inferred
    from an unrelated checkout or replaced by the installed runtime revision.
    """

    def __init__(
        self,
        params: Sequence[ParamInfo],
        *,
        platform: str,
        runtime: str,
        sdk_revision: str,
        return_aliases: Sequence[int] = (),
    ) -> None:
        if platform not in ("a2a3", "a5"):
            raise ValueError(f"Unsupported kernel platform {platform!r}")
        if runtime not in ("host_build_graph", "tensormap_and_ringbuffer"):
            raise ValueError(f"Unsupported kernel runtime {runtime!r}")
        if type(sdk_revision) is not str or re.fullmatch(r"[0-9a-f]{40}", sdk_revision) is None:
            raise ValueError(f"Kernel SDK revision must be a full git revision, got {sdk_revision!r}")
        self._params = tuple(
            ParamInfo(p.name, p.direction, list(p.shape) if p.shape is not None else None, p.dtype, p.layout)
            for p in params
        )
        self._platform, self._runtime, self._sdk_revision = platform, runtime, sdk_revision
        self._return_aliases = tuple(return_aliases)
        names = [p.name for p in self._params]
        if any(type(name) is not str or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Kernel parameter names must be nonempty and unique")
        for p in self._params:
            if p.direction not in (ParamDirection.In, ParamDirection.Out, ParamDirection.InOut):
                raise ValueError(f"Invalid direction for kernel parameter {p.name!r}: {p.direction}")
            if p.shape is None:
                if (
                    p.direction != ParamDirection.In
                    or str(p.dtype) not in _KERNEL_SCALARS
                    or p.layout is not None
                ):
                    raise ValueError(
                        f"Unsupported kernel scalar {p.name!r}: {p.dtype}/{p.direction}/{p.layout}"
                    )
            else:
                if _to_torch_dtype(p.dtype) is None:
                    raise ValueError(f"Unsupported kernel tensor dtype {p.dtype} for {p.name!r}")
                if p.layout not in (None, "ND", "DN", "NZ"):
                    raise ValueError(f"Unsupported kernel layout {p.layout!r} for {p.name!r}")
                if not 1 <= len(p.shape) <= MAX_KERNEL_RANK or any(
                    type(dim) is not int or not (dim == -1 or 0 < dim < 2**32) for dim in p.shape
                ):
                    raise ValueError(
                        f"Kernel parameter {p.name!r} requires rank 1..5 and positive u32/-1 dimensions"
                    )
        if len(self.tensor_indices) > MAX_KERNEL_TENSORS or len(self.scalar_indices) > MAX_KERNEL_SCALARS:
            raise ValueError("Kernel signature exceeds L2 argument capacity")
        for index in self._return_aliases:
            if (
                type(index) is not int
                or not 0 <= index < len(self._params)
                or self._params[index].shape is None
            ):
                raise ValueError(f"Return alias must reference an external tensor parameter, got {index!r}")

    @property
    def parameters(self) -> tuple[ParamInfo, ...]:
        """Return copies so callers cannot mutate the descriptor's signature."""
        return tuple(
            ParamInfo(p.name, p.direction, list(p.shape) if p.shape is not None else None, p.dtype, p.layout)
            for p in self._params
        )

    @property
    def tensor_indices(self) -> tuple[int, ...]:
        """Original signature positions, in L2 tensor-pool order."""
        return tuple(i for i, p in enumerate(self._params) if p.shape is not None)

    @property
    def scalar_indices(self) -> tuple[int, ...]:
        """Original signature positions, in L2 scalar-pool order."""
        return tuple(i for i, p in enumerate(self._params) if p.shape is None)

    def bind(self, args: Sequence[_Arg]) -> tuple[list[_Arg], list[_Arg], list[_Arg]]:
        """Bind all external arguments without allocation, conversion or native imports."""
        bound = bind_complete_args(args, self._params, caller_name="Kernel call")
        return (
            [bound[i] for i in self.tensor_indices],
            [bound[i] for i in self.scalar_indices],
            [bound[i] for i in self._return_aliases],
        )

    def record(self) -> dict[str, Any]:
        """Derive a canonical artifact descriptor from the common signature."""
        return {
            "schema": 1,
            "argument_abi": "ChipStorageTaskArgs",
            "platform": self._platform,
            "runtime": self._runtime,
            "sdk_revision": self._sdk_revision,
            "params": [
                _param_info_to_dict(p) | {"shape": list(p.shape) if p.shape is not None else None}
                for p in self._params
            ],
            "return_aliases": list(self._return_aliases),
        }

    @classmethod
    def from_record(cls, record: Any) -> "KernelABI":
        """Reject malformed or noncanonical descriptors without loading a runtime."""
        if not isinstance(record, dict):
            raise ValueError("Kernel ABI descriptor must be an object; recompile")
        record = cast(dict[str, Any], record)
        try:
            params = [_param_info_from_dict(p) for p in record["params"]]
            result = cls(
                params,
                platform=record["platform"],
                runtime=record["runtime"],
                sdk_revision=record["sdk_revision"],
                return_aliases=record["return_aliases"],
            )
            if encode_record(record) != encode_record(result.record()):
                raise ValueError("descriptor is not canonical")
            return result
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid kernel ABI descriptor: {exc}; recompile") from exc

    def require_compatible(self, expected: "KernelABI") -> None:
        """Check signature, target and SDK identity before loading an artifact."""
        if encode_record(self.record()) != encode_record(expected.record()):
            raise ValueError("Kernel artifact ABI/signature or SDK does not match; recompile")
