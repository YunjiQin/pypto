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
from dataclasses import KW_ONLY, dataclass
from typing import Any, cast

from pypto._identity import encode_record
from pypto.ir.param_info import (
    ParamInfo,
    _param_info_from_dict,
    _param_info_to_dict,
)

# Shared L2 limits, checked against the pinned SDK by the host-only ABI UT.
MAX_KERNEL_TENSORS = 256
MAX_KERNEL_SCALARS = 128


@dataclass(frozen=True)
class KernelABI:
    """Snapshot a common signature and its kernel-specific return contract.

    SDK identity must come from the compilation context. It is never inferred
    from an unrelated checkout or replaced by the installed runtime revision.
    """

    parameters: Sequence[ParamInfo]
    _: KW_ONLY
    platform: str
    runtime: str
    sdk_revision: str
    return_aliases: Sequence[int] = ()

    def __post_init__(self) -> None:
        if self.platform not in ("a2a3", "a5"):
            raise ValueError(f"Unsupported kernel platform {self.platform!r}")
        if self.runtime not in ("host_build_graph", "tensormap_and_ringbuffer"):
            raise ValueError(f"Unsupported kernel runtime {self.runtime!r}")
        if type(self.sdk_revision) is not str or re.fullmatch(r"[0-9a-f]{40}", self.sdk_revision) is None:
            raise ValueError(f"Kernel SDK revision must be a full git revision, got {self.sdk_revision!r}")
        # Frozen fields must also own immutable sequences, so later mutations
        # to the caller's lists cannot change the artifact identity.
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "return_aliases", tuple(self.return_aliases))
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("Kernel parameter names must be unique")
        tensor_count = sum(p.shape is not None for p in self.parameters)
        if tensor_count > MAX_KERNEL_TENSORS or len(self.parameters) - tensor_count > MAX_KERNEL_SCALARS:
            raise ValueError("Kernel signature exceeds L2 argument capacity")
        for index in self.return_aliases:
            if (
                type(index) is not int
                or not 0 <= index < len(self.parameters)
                or self.parameters[index].shape is None
            ):
                raise ValueError(f"Return alias must reference an external tensor parameter, got {index!r}")

    def record(self) -> dict[str, Any]:
        """Derive a canonical artifact descriptor from the common signature."""
        return {
            "schema": 1,
            "argument_abi": "ChipStorageTaskArgs",
            "platform": self.platform,
            "runtime": self.runtime,
            "sdk_revision": self.sdk_revision,
            "params": [_param_info_to_dict(p) for p in self.parameters],
            "return_aliases": list(self.return_aliases),
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
        if self != expected:
            raise ValueError("Kernel artifact ABI/signature or SDK does not match; recompile")
