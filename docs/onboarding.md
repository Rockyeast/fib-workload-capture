# Minimal model onboarding

The library captures and exports operator workloads. A caller or Skill owns model selection,
GPU/TP settings, engine startup, requests, and iterative definition review.

```python
from fib_workload_capture import capture, export

with capture("runs/new-capture", definitions="reviewed-definitions",
             inventory="runs/probe/inventory.json", metadata={"versions": versions}):
    # Import/start the engine here so spawned workers inherit capture settings.
    engine = start_engine()
    try:
        send_requests(engine)
    finally:
        engine.shutdown()

report = export("runs/new-capture", max_workloads=20)
assert report["accepted"]
```

Omit `definitions` to probe native definitions and executed SGLang modules. Use a fresh output
directory and Python process for each pass. Definitions may be manually or agent-authored;
review the raw templates, then call `checks.check_definitions()` before workload collection.

A complete runnable SGLang example is `examples/sglang_capture.py`:

```bash
python examples/sglang_capture.py --model /path/to/model \
  --run runs/probe --engine-config engine.json --requests requests.json
python examples/sglang_capture.py --model /path/to/model \
  --run runs/collect --engine-config engine.json --requests requests.json \
  --definitions reviewed-definitions --inventory runs/probe/inventory.json
```

`engine.json` contains caller-chosen SGLang kwargs (including `disable_cuda_graph: true`).
`requests.json` is a list of `{ "input_ids": [[1, 2, 3]], "sampling_params":
{ "temperature": 0, "max_new_tokens": 8, "ignore_eos": true } }` objects. Use actual tokenizer
IDs appropriate to the model. The example records both files' contents in run metadata.

Outputs: `definitions/`, `metadata.json`, `capture.json`, `raw/`, `output/`, `report.json`.
Raw captures remain available for diagnosis. Remove only a run's generated `raw/` after
checking its output; the API never silently overwrites a previous run/output directory.

`accepted` means valid dataset structure, nonempty workloads for every reviewed definition,
and no unmatched observed calls in selected SGLang module instances or reported worker errors.
It does **not** imply all-model coverage or numerical correctness. Native FlashInfer coverage
currently checks definition presence, not every native call. Numerical replay is separately
reported as `not_run`; use existing reference/benchmark tools for that validation.

The `onboarding dump-definition/dump-workload --agent` CLI and bundled serving planner from
flashinfer-bench PR #428 are not included. Definitions and dataset output use the standard
FlashInfer Trace format. Workflow
instructions live in `.claude/skills/onboard-model/SKILL.md`; no agent subprocess is started
by the library. Request generation and deployment are not duplicated in this package.

## vLLM: native FlashInfer APIs

Use `examples/vllm_capture.py` with caller-chosen vLLM `LLM` kwargs and
`enforce_eager: true`. Its request format matches the SGLang example, but sampling parameters
use vLLM names such as `max_tokens`. Select an actual FlashInfer execution path in the engine
configuration; installing FlashInfer alone does not guarantee it is called.

```bash
python examples/vllm_capture.py --model /path/to/model \
  --run runs/vllm-probe --engine-config vllm.json --requests requests.json
python examples/vllm_capture.py --model /path/to/model \
  --run runs/vllm-collect --engine-config vllm.json --requests requests.json \
  --definitions reviewed-fi-definitions
```

The example uses `capture(..., backend="flashinfer")`: native definitions on the first pass,
then native input dumps and standard export on the second. This mode rejects SGLang tags and
does not produce a SGLang module inventory. vLLM's non-FlashInfer operators are unsupported.
The normal `backend="sglang"` default remains unchanged. Both modes reuse the same exporter.
Set `FLASHINFER_DUMP_MAX_COUNT` and `FLASHINFER_DUMP_MAX_SIZE_GB` before running to bound dumps;
a limit can truncate captures, so nonempty definitions are not proof of all-call coverage.

Native-only exports use strict input binding: wrapper `plan()` data is paired with
`run()` by process, wrapper class, and instance; nested K/V tensors retain their
observed layout. Tensor values (including gates and structural indices) are saved,
and missing required inputs or shape/dtype mismatches are rejected. Safetensors
stores contiguous values, so original strides are not reproduced in the dataset.
Review the native templates before collection: a generated template can omit
last-page lengths or describe conceptual GDN parameters instead of actual API inputs.
Capture limits are global in the tested FlashInfer logger; warmup can exhaust a
small limit before decode. Inspect recorded API counts rather than assuming every
requested phase was captured. Schema acceptance remains separate from numerical replay.

## Run the exported workloads

The export is a standard TraceSet. Add reviewed `Solution` JSON files under
`output/solutions/<op_type>/<definition>/`, then use the existing benchmark runner:

```bash
python -m flashinfer_bench run --local runs/collect/output \
  --definitions YOUR_DEFINITION --solutions YOUR_SOLUTION \
  --warmup-runs 2 --iterations 5 --num-trials 2 \
  --required-matched-ratio 1.0 --no-profile-baseline
python -m flashinfer_bench report summary --local runs/collect/output
```

Set numerical tolerances from the operator's contract before running. The runner loads
captured safetensors, executes the definition's reference and the supplied solution,
and saves correctness and timing results in `traces/`. A definition by itself is not a
solution: missing solutions are skipped, so check the expected trace count as well as
individual `PASSED` statuses. `--no-profile-baseline` skips reference timing, not reference
correctness; it produces no meaningful speedup comparison.

For a wrapper-based solution, distinguish `plan()` plus `run()` timing from a prepared
`run()`-only kernel measurement. Do not present this functional smoke as model throughput.

Before claiming coverage, compare raw API counts/shapes with exported workload axes and
inspect `raw/flashinfer_api.log` for count/size-limit warnings. The request list records
what was requested, not what survived capture limits and export selection. Warmup calls
can be included. Prefer shorter generation, per-scenario runs, or a smaller reviewed
API set before increasing dump budgets. Keep unresolved numerical cases explicitly
unverified; do not adjust their inputs or tolerances to make the capture appear valid.
