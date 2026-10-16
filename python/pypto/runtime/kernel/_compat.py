# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Checks for the audited Simpler kernel boundary, without initializing workers.

Registration uses the Python ChipWorker and returns its opaque handle. Native
borrowed initialization and caller-stream submission are separate capabilities;
this module does not claim that the Torch launch adapter is already integrated.
"""

import importlib
from dataclasses import dataclass
from typing import Any

from pypto.runtime.runtime_pin import RuntimePinMismatch, check_runtime_pin

# Reviewed against the native signatures and lifetime contract at this revision.
# A new revision needs a contract review, not just a renamed method probe.
_AUDITED_REVISION = "c1f89ea431f5306f78c0e4ab9573c829756d7f1e"


@dataclass(frozen=True)
class KernelCapabilities:
    """Independent runtime capabilities; no device state is acquired."""

    handle_registration: bool
    native_eager: bool
    diagnostics: bool
    graph_capture: bool = False

    def require(self, *, capture: bool = False, diagnostics: bool = False) -> None:
        """Reject only the requested unsupported operation."""
        if capture and not self.graph_capture:
            raise NotImplementedError("Torch kernel graph capture is not supported by this runtime")
        if not self.handle_registration:
            raise NotImplementedError("Torch kernels require Simpler Python CallableHandle registration")
        if not self.native_eager:
            raise NotImplementedError(
                "Torch kernels require borrowed-context init and caller-stream submission"
            )
        if diagnostics and not self.diagnostics:
            raise NotImplementedError(
                "Torch kernel diagnostics require per-run config and retained-run flush"
            )


def _methods(owner: Any, *names: str) -> bool:
    return all(callable(getattr(owner, name, None)) for name in names)


def inspect_kernel_capabilities(sdk_revision: str) -> KernelCapabilities:
    """Check installed native identity against the compilation SDK before probing.

    Optional dependency errors and native exceptions keep their original types.
    No catch-all converts an installation/ABI failure to NotImplementedError.
    The legacy single-call supported bit has no bearing on native eager support.
    """
    check_runtime_pin()
    native = importlib.import_module("_task_interface")
    actual = getattr(native, "__build_commit__", None)
    if not sdk_revision or actual != sdk_revision:
        raise RuntimePinMismatch(
            f"Kernel compilation SDK {sdk_revision!r} differs from native binding {actual!r}"
        )
    if actual != _AUDITED_REVISION:
        raise RuntimeError(f"Kernel runtime contract has not been reviewed for SDK revision {actual!r}")
    python = importlib.import_module("simpler.task_interface")
    worker = getattr(native, "_ChipWorker", None)
    run = getattr(native, "_ChipRun", None)
    wrapper = getattr(python, "ChipWorker", None)
    registration = _methods(wrapper, "register_callable", "unregister_callable")
    eager = _methods(
        worker, "init", "register_callable", "_submit_chip_run_direct", "_close_chip_run_lane", "finalize"
    ) and _methods(run, "done", "wait")
    config = getattr(native, "CallConfig", None)
    diagnostics = _methods(worker, "flush_diagnostics") and all(
        hasattr(config, name) for name in ("enable_chip_swimlane", "enable_dep_gen")
    )
    return KernelCapabilities(registration, eager, diagnostics)
