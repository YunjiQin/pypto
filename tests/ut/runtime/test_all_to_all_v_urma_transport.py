# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execute the builtin engine helpers with a release-mode URMA test double."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module", params=["mte", "urma"])
def engine_probe(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> tuple[Path, str]:
    engine = getattr(request, "param")
    assert isinstance(engine, str)
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("engine probe requires a C++17 host compiler")
    root = Path(__file__).resolve().parents[3]
    template = root / "python/pypto/runtime/builtins/collectives/all_to_all_v/templates/kernel.cpp.in"
    text = template.read_text().replace("{{engine_urma}}", str(int(engine == "urma")))
    begin = text.index("template <typename T>\nAICORE inline __gm__ T *CommRemotePtr")
    end = text.index("// N-way credit barrier", begin)
    folder = tmp_path_factory.mktemp("urma_engine")
    source = folder / "probe.cpp"
    source.write_text(
        r"""
#include <cstdint>
#include <cstdlib>
#include <cstring>
#define AICORE
#define __gm__
#define PTO_URMA_SUPPORTED
#define USE_URMA ENGINE_URMA
#define PTO_ASSERT(condition, message) ((void)0)
struct CommContext { uint64_t windowsIn[2]; uint64_t workSpace; uint32_t rankId; };
int posts = 0, waits = 0, mte = 0;
bool session_ok = true, event_ok = true, completion_ok = true;
uint64_t registered_base = 0;
void trap() { std::exit(42); }
namespace pto::comm {
enum class DmaEngine { URMA };
struct AsyncSession {};
namespace urma {
uint64_t UrmaPeerMrBaseAddr(uint8_t *, uint32_t) { return registered_base; }
}
template <DmaEngine> bool BuildAsyncSession(uint8_t *workspace, uint32_t, AsyncSession &) {
    return session_ok && workspace;
}
struct Event {
    int *dst;
    int *src;
    bool valid() const { return event_ok; }
    bool Wait(const AsyncSession &) const {
        ++waits;
        if (completion_ok) *dst = *src;
        return completion_ok;
    }
};
template <DmaEngine> Event TPUT_ASYNC(int &dst, int &src, const AsyncSession &, uint32_t) {
    ++posts;
    return {&dst, &src};
}
void TPUT(int &dst, int &src, int &) { ++mte; dst = src; }
}
"""
        + text[begin:end]
        + r"""
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    int local[8] = {}, ipc[8] = {}, registered[8] = {};
    CommContext ctx{{reinterpret_cast<uint64_t>(local), reinterpret_cast<uint64_t>(ipc)}, 1, 0};
    registered_base = reinterpret_cast<uint64_t>(registered);
    const char *mode = argv[1];
    if (!std::strcmp(mode, "address"))
        return PayloadRemotePtr(&ctx, local + 3, 1) == (USE_URMA ? registered : ipc) + 3 ? 0 : 2;
    if (!std::strcmp(mode, "unregistered")) {
        registered_base = 0;
        PayloadRemotePtr(&ctx, local + 3, 1);
        return 3;
    }
    if (!std::strcmp(mode, "no_workspace")) ctx.workSpace = 0;
    const int peer = !std::strcmp(mode, "self") ? 0 : 1;
    session_ok = std::strcmp(mode, "bad_session");
    event_ok = std::strcmp(mode, "bad_event");
    completion_ok = std::strcmp(mode, "bad_completion");
    int src = 73, dst = 0, tile = 0;
    PushPayload(dst, src, tile, &ctx, peer);
    const bool fallback = !USE_URMA || peer == 0;
    return dst == src && (fallback ? mte == 1 && posts == 0 && waits == 0
                                  : mte == 0 && posts == 1 && waits == 1) ? 0 : 4;
}
"""
    )
    binary = folder / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            f"-DENGINE_URMA={int(engine == 'urma')}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary, engine


@pytest.mark.parametrize(
    ("mode", "returncode"),
    [
        ("address", 0),
        ("remote", 0),
        ("no_workspace", 0),
        ("self", 0),
        ("unregistered", 42),
        ("bad_session", 42),
        ("bad_event", 42),
        ("bad_completion", 42),
    ],
)
def test_release_engine_contract(engine_probe: tuple[Path, str], mode: str, returncode: int) -> None:
    binary, engine = engine_probe
    if engine == "mte":
        returncode = 3 if mode == "unregistered" else 0
    elif mode == "no_workspace":
        returncode = 42
    result = subprocess.run([str(binary), mode], check=False)
    assert result.returncode == returncode, mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
