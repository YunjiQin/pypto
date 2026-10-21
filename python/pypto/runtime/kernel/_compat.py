# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check the public runtime contracts without probing a device or old kernel bits."""

import importlib
import inspect
from types import SimpleNamespace
from typing import Any

from pypto.runtime.runtime_pin import check_runtime_pin


def _require_method(owner: Any, name: str, label: str) -> None:
    if not callable(getattr(owner, name, None)):
        raise NotImplementedError(f"Torch kernel runtime requires {label}.{name}")


def require_runtime() -> Any:
    """Preserve the shared pin guard, then check the installed public interface.

    This checks interface availability, not device execution. Native identity is
    checked by check_runtime_pin under its existing missing/skip rules.
    """
    check_runtime_pin()
    simpler = importlib.import_module("simpler")
    interface = importlib.import_module("simpler.task_interface")
    workers = importlib.import_module("simpler.worker")
    worker = getattr(simpler, "Worker", None)
    for name in ("init", "register", "unregister", "submit", "close", "flush_diagnostics"):
        _require_method(worker, name, "Worker")
    for method, keyword in (("init", "borrowed_context_generation"), ("submit", "caller_stream")):
        if keyword not in inspect.signature(getattr(worker, method)).parameters:
            raise NotImplementedError(f"Torch kernel runtime requires Worker.{method}(..., {keyword}=...)")
    if not hasattr(worker, "device_location") or not hasattr(getattr(workers, "RunHandle", None), "done"):
        raise NotImplementedError("Torch kernel runtime requires Worker.device_location and RunHandle.done")
    for owner, label, methods in (
        (getattr(workers, "RunHandle", None), "RunHandle", ("wait",)),
        (getattr(interface, "Buffer", None), "Buffer", ("wrap", "close")),
        (getattr(interface, "TaskArgs", None), "TaskArgs", ("add_tensor", "add_scalar")),
        (getattr(interface, "CallConfig", None), "CallConfig", ("validate",)),
    ):
        for method in methods:
            _require_method(owner, method, label)
    if not callable(getattr(interface, "Tensor", None)):
        raise NotImplementedError("Torch kernel runtime requires Tensor views")
    return SimpleNamespace(Worker=worker, CallConfig=interface.CallConfig)
