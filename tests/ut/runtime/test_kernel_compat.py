# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime guards inspect Python registration and native execution independently."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pypto.runtime.kernel import _compat
from pypto.runtime.runtime_pin import RuntimePinMismatch

SDK = "c1f89ea431f5306f78c0e4ab9573c829756d7f1e"


@pytest.fixture
def runtime(monkeypatch):
    wrapper = SimpleNamespace(register_callable=Mock(), unregister_callable=Mock())
    worker = SimpleNamespace(
        **{
            name: Mock()
            for name in (
                "init",
                "register_callable",
                "_submit_chip_run_direct",
                "_close_chip_run_lane",
                "finalize",
                "flush_diagnostics",
            )
        }
    )
    native = SimpleNamespace(
        __build_commit__=SDK,
        _ChipWorker=worker,
        _ChipRun=SimpleNamespace(done=Mock(), wait=Mock()),
        CallConfig=SimpleNamespace(enable_chip_swimlane=0, enable_dep_gen=False),
        simpler_kernel_mode_supported=lambda: 0,
    )
    modules = {"_task_interface": native, "simpler.task_interface": SimpleNamespace(ChipWorker=wrapper)}
    monkeypatch.setattr(_compat, "check_runtime_pin", Mock())
    monkeypatch.setattr(_compat.importlib, "import_module", lambda name: modules[name])
    return SimpleNamespace(native=native, worker=worker, wrapper=wrapper)


def test_new_interfaces_pass_without_legacy_methods(runtime):
    caps = _compat.inspect_kernel_capabilities(SDK)
    caps.require(diagnostics=True)
    assert caps.handle_registration and caps.native_eager and not caps.graph_capture
    runtime.wrapper.register_callable.assert_not_called()
    runtime.wrapper.unregister_callable.assert_not_called()
    for method in vars(runtime.worker).values():
        method.assert_not_called()
    with pytest.raises(NotImplementedError, match="capture"):
        caps.require(capture=True)


@pytest.mark.parametrize(
    "missing,capability",
    [
        ("init", "native_eager"),
        ("_submit_chip_run_direct", "native_eager"),
        ("_close_chip_run_lane", "native_eager"),
        ("flush_diagnostics", "diagnostics"),
    ],
)
def test_independent_missing_capabilities(runtime, missing, capability):
    delattr(runtime.worker, missing)
    caps = _compat.inspect_kernel_capabilities(SDK)
    assert not getattr(caps, capability)
    if capability == "diagnostics":
        caps.require()
    with pytest.raises(NotImplementedError):
        caps.require(diagnostics=True)


def test_missing_per_run_diagnostics_keeps_eager_available(runtime):
    del runtime.native.CallConfig.enable_dep_gen
    caps = _compat.inspect_kernel_capabilities(SDK)
    caps.require()
    with pytest.raises(NotImplementedError, match="diagnostics"):
        caps.require(diagnostics=True)


def test_missing_python_wrapper_is_not_replaced_by_native_ids(runtime):
    del runtime.wrapper.register_callable
    with pytest.raises(NotImplementedError, match="CallableHandle"):
        _compat.inspect_kernel_capabilities(SDK).require()


@pytest.mark.parametrize("actual", [None, "", "a" * 40])
def test_sdk_native_mismatch_is_not_unsupported(runtime, actual):
    runtime.native.__build_commit__ = actual
    with pytest.raises(RuntimePinMismatch, match="compilation SDK"):
        _compat.inspect_kernel_capabilities(SDK)


def test_unknown_matching_revision_is_not_claimed_supported(runtime):
    runtime.native.__build_commit__ = "b" * 40
    with pytest.raises(RuntimeError, match="not been reviewed"):
        _compat.inspect_kernel_capabilities("b" * 40)


def test_pin_failure_keeps_original_exception(runtime, monkeypatch):
    failure = RuntimePinMismatch("installed pin mismatch")
    monkeypatch.setattr(_compat, "check_runtime_pin", Mock(side_effect=failure))
    with pytest.raises(RuntimePinMismatch) as caught:
        _compat.inspect_kernel_capabilities(SDK)
    assert caught.value is failure


def test_import_failure_keeps_original_exception(runtime, monkeypatch):
    failure = ImportError("broken native binding")
    monkeypatch.setattr(_compat.importlib, "import_module", Mock(side_effect=failure))
    with pytest.raises(ImportError) as caught:
        _compat.inspect_kernel_capabilities(SDK)
    assert caught.value is failure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
