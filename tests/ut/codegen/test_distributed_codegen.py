# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for distributed Python code generation."""

import pypto.language as pl
import pytest
from pypto import codegen, passes


class TestDistributedCodegen:
    """Test distributed Python codegen on outlined hierarchy programs."""

    def test_chip_worker_and_orchestrator(self):
        """HOST orchestrator calling CHIP worker produces submit_next_level."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Worker)
            def chip_worker(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = self.chip_worker(x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Verify imports
        assert "from simpler.task_interface import TaskArgs, TensorArgType, make_tensor_arg" in code

        # Verify function definition
        assert "def host_orch" in code
        assert "orch, _args, config" in code

        # Verify call-site lowering: CHIP worker → submit_next_level
        assert "submit_next_level" in code
        assert 'callables["chip_worker"]' in code
        assert "TaskArgs()" in code

    def test_sub_worker_submit_sub(self):
        """HOST worker (SubWorker) produces submit_sub call."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def verify(self, f: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                self.verify(x)
                return x

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # HOST worker (level 3) → submit_sub
        assert "submit_sub" in code
        assert 'sub_ids["verify"]' in code

    def test_chip_and_sub_worker_combined(self):
        """Program with both CHIP worker and SubWorker."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Worker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def verify(self, f: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                f: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, b)
                self.verify(f)
                return f

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "submit_next_level" in code
        assert "submit_sub" in code
        assert "TensorArgType.INPUT" in code

    def test_for_loop_codegen(self):
        """ForStmt in function body produces Python for loop."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.POD, role=pl.Role.Orchestrator)
            def orch_with_loop(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = x
                for i in pl.range(0, 4):
                    y = pl.add(y, x)
                return y

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "for " in code
        assert "in range(" in code

    def test_python_imports(self):
        """Generated code contains required Python imports."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def simple_worker(self, x: pl.Tensor[[64], pl.FP32]):
                pass

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "from simpler.task_interface import TaskArgs, TensorArgType, make_tensor_arg" in code

    def test_tensor_arg_type_tags(self):
        """Parameter directions map to correct TensorArgType tags."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Worker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                out: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, b, f)
                return out

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "TensorArgType.INPUT" in code
        assert "TensorArgType.OUTPUT_EXISTING" in code

    def test_bool_constants(self):
        """Boolean constants use Python True/False, not C++ true/false."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def worker(self, x: pl.Tensor[[64], pl.FP32]):
                pass

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Python uses True/False, not true/false
        assert "true" not in code.lower() or "True" in code or "False" in code

    def test_sub_worker_pure_python_body(self):
        """HOST Worker with pure Python body is captured without DSL parsing."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def verify(self, f: pl.Tensor[[128, 128], pl.FP32]):
                import torch  # noqa: PLC0415

                expected = torch.full((128, 128), 5.0, dtype=torch.float32)
                assert torch.allclose(f, expected)  # pyright: ignore[reportArgumentType]

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
                self.verify(x)
                return x

        # Should not raise — pure Python body is skipped during DSL parsing
        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        assert "submit_sub" in code
        assert 'sub_ids["verify"]' in code

    def test_sub_worker_callable_captured(self):
        """HOST Worker callable is captured in the sub_worker registry."""
        from pypto.language.parser.decorator import (  # noqa: PLC0415
            get_sub_worker_callables,  # pyright: ignore[reportAttributeAccessIssue]
        )

        @pl.program
        class Input:
            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def verify(self, f: pl.Tensor[[64], pl.FP32]):
                pass

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                self.verify(x)
                return x

        subs = get_sub_worker_callables(Input)
        assert "verify" in subs
        assert isinstance(subs["verify"], str)

    def test_create_tensor_emits_shared_torch_zeros(self):
        """tensor.create in HOST orchestrator emits torch.zeros(...).share_memory_()."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Worker)
            def chip_worker(
                self,
                a: pl.Tensor[[64], pl.FP32],
                buf: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, a)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                buf: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.chip_worker(a, buf)
                return result

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # torch.zeros with share_memory_() emitted
        assert "torch.zeros(" in code
        assert "torch.float32" in code
        assert ".share_memory_()" in code
        assert "import torch" in code

    def test_create_tensor_shared_zeros_for_multiple_tensors(self):
        """Multiple tensor.create calls each emit torch.zeros(...).share_memory_()."""

        @pl.program
        class Input:
            @pl.function(level=pl.Level.CHIP, role=pl.Role.Worker)
            def chip_add(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
                return y

            @pl.function(level=pl.Level.CHIP, role=pl.Role.Worker)
            def chip_sub(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.sub(a, b)
                return y

            @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
            def reduce_sum(
                self,
                sum_ab: pl.Tensor[[64], pl.FP32],
                diff_ab: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return f

            @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
            def host_orch(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                f: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                sum_ab: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                diff_ab: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                out_sum: pl.Tensor[[64], pl.FP32] = self.chip_add(a, b, sum_ab)
                out_diff: pl.Tensor[[64], pl.FP32] = self.chip_sub(a, b, diff_ab)
                out_f: pl.Tensor[[64], pl.FP32] = self.reduce_sum(out_sum, out_diff, f)
                return out_f

        program = passes.convert_to_ssa()(Input)
        cg = codegen.DistributedCodegen()
        code = cg.generate(program)

        # Two torch.zeros().share_memory_() calls
        assert code.count("torch.zeros(") == 2
        assert code.count(".share_memory_()") == 2
        # Parameter tensors still use make_tensor_arg(tensors[...])
        assert 'make_tensor_arg(tensors["a' in code
        assert 'make_tensor_arg(tensors["b' in code


class TestSubWorkerSourceGeneration:
    """Test _generate_sub_worker_source for correct param names and imports."""

    def test_sub_worker_source_uses_original_param_names(self):
        """_user_* function params use original names, not SSA-renamed ones."""
        from pypto.backend.pto_backend import _generate_sub_worker_source  # noqa: PLC0415

        body = "assert torch.allclose(f, expected)\n"
        source = _generate_sub_worker_source("verify", body, ["f__ssa_v0"])

        # _user_verify should use original name 'f'
        assert "def _user_verify(f):" in source
        # wrapper should unpack with SSA name and call _user with SSA value
        assert "f__ssa_v0 = _tensor_from_continuous(args.tensor(0))" in source
        assert "_user_verify(f__ssa_v0)" in source

    def test_sub_worker_source_imports_torch(self):
        """Generated SubWorker source includes import torch."""
        from pypto.backend.pto_backend import _generate_sub_worker_source  # noqa: PLC0415

        source = _generate_sub_worker_source("worker", "pass\n", ["x__ssa_v0"])
        assert "import torch" in source

    def test_sub_worker_source_multiple_params(self):
        """Multiple SSA params all stripped correctly."""
        from pypto.backend.pto_backend import _generate_sub_worker_source  # noqa: PLC0415

        body = "result = torch.add(sum_ab, diff_ab)\nf[:] = result\n"
        source = _generate_sub_worker_source(
            "reduce_sum",
            body,
            ["sum_ab__ssa_v0", "diff_ab__ssa_v0", "f__ssa_v0"],
        )

        assert "def _user_reduce_sum(sum_ab, diff_ab, f):" in source
        assert "_user_reduce_sum(sum_ab__ssa_v0, diff_ab__ssa_v0, f__ssa_v0)" in source

    def test_sub_worker_source_no_suffix_passthrough(self):
        """Param without auto-name suffix passes through unchanged."""
        from pypto.backend.pto_backend import _generate_sub_worker_source  # noqa: PLC0415

        source = _generate_sub_worker_source("worker", "pass\n", ["x"])
        assert "def _user_worker(x):" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
