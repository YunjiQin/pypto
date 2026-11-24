# Best measured EP2 indexed dispatch snapshot

This branch preserves the simpler implementation tested on 2026-11-02. It is
not a PyPTO builtin integration. Source is in
`tests/st/runtime/indexed_dispatch_snapshot/` as an overlay for simpler, not a
standalone PyPTO pytest suite. Original simpler module imports are retained.

## Results

One device reservation on A5 devices 0/3 ran A, B, then A again. All use EP2,
72 tokens/rank, H=5120, TopK=6, three experts/rank, 32 AIVs, URMA, five warmups
and 100 samples. B passed 18 scenes and full benchmark output validation.

| Mean microseconds | A | B | A repeat |
| --- | ---: | ---: | ---: |
| BF16 | 48.814 | 45.655 | 48.355 |
| FP8 | 40.800 | 39.315 | 40.807 |

B reduces mean latency by 6.03%/3.65% relative to the two A runs. B p95 is
46.817/40.503 us. Timing covers device dispatch including routing, communication,
histogram/prefix and expansion; it excludes host staging, task launch,
round-start reuse barrier, and quantization. FP8 input and scales are prebuilt.
Stage maxima overlap and cannot be added to obtain the total.

## Implementation

Send indices deduplicate each token by destination rank. SIMT builds URMA SQEs
with hidden/metadata SGE pairs and requests only the final CQE. Receivers divide
work by source rank and contiguous token interval. Each AIV bulk-loads route IDs,
computes a local histogram, acquires all histogram-ready flags through batched
MTE reads, then computes its own aligned expert offsets. Only AIV 0 writes
output counts; no prefix-ready publication or cursor array is needed. Hidden
uses an eight-slot MTE ring; metadata uses cooperative warp copies.

The kernel is byte-identical to measured B, SHA256
`4131608ef5e077dd598bf97ebbf6f8a3d1ff4655e7dfaa9cb802b73e81e3b205`.
Later 12-expert and vector-histogram experiments are excluded. The host harness
is restored to control stride 512 and restricted to three experts. JSON files
preserve measurement provenance; the manifest also records packaged hashes.
Historical hardware results are not a fresh run at publication time.

## Reproduce

Use a separate simpler checkout at
`fb63f4c02220ed99a70150409c676711dcdbb418`, with PTO-ISA pin
`217bb98952a8f9004b95fe938da97d8d5f68f702` and CANN 9.3.0/A5 URMA support.
Build/install simpler in that checkout's local virtual environment following
its own build instructions. Do not copy binaries between checkouts.

Copy this snapshot's `reference.py`, `run.py.in`, `test_indexed_dispatch.py.in`, and
`kernels/` into simpler's
`examples/a5/tensormap_and_ringbuffer/indexed_dispatch/`, removing the `.in`
suffix from the two Python harness files. The suffix prevents accidental
collection by PyPTO CI. From that simpler
checkout, activate its venv and run:

```bash
python -m examples.a5.tensormap_and_ringbuffer.indexed_dispatch.run --compile-only
```

Run the scene directory with simpler's A5 test runner through `task-submit`,
using two reserved devices. For performance, within the same reservation:

```bash
python -m examples.a5.tensormap_and_ringbuffer.indexed_dispatch.run \
  --nranks 2 --tokens 72 --local-experts 3 --transport urma --dtype fp8 \
  --warmup 5 --samples 100 --output outputs/indexed-dispatch-fp8.json
```

The runner reads allocated device IDs from `TASK_DEVICE`. Repeat with
`--dtype bf16`. Communication domains use `get_or_allocate_domain` and persist
within the Worker; Worker close releases them. EP8 performance is not validated
for this snapshot.

The CPU oracle tests can also run directly in this fork with torch and pytest:

```bash
python -m pytest tests/st/runtime/indexed_dispatch_snapshot/test_reference.py \
  --confcutdir=tests/st/runtime/indexed_dispatch_snapshot -v
```
