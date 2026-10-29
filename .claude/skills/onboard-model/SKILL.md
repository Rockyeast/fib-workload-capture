---
name: onboard-model
description: Collect real model operator workloads for FlashInfer-Bench, review definitions against executed code, and diagnose capture gaps. Use for model onboarding; model deployment remains caller-owned.
---

# Onboard a model

Use the small API in `fib_workload_capture`: `capture()` surrounds caller-owned
inference; `export()` converts captured inputs and reports coverage. The library does not
choose GPU counts, deploy servers, call another agent, or submit PRs.

1. Inspect the installed serving engine and available resources. Choose model path, TP,
   backend, and a bounded request set appropriate to the task. Record the actual engine
   options, package versions, and exact token IDs. Prefer an existing working launch command.
2. For a new model, run without reviewed definitions to obtain native definitions and
   `inventory.json`. The minimal executable example is
   `examples/sglang_capture.py`; its engine options and requests are ordinary JSON.
   For vLLM use `examples/vllm_capture.py` and `backend="flashinfer"`: this captures
   native FlashInfer APIs only, without a non-FlashInfer module inventory. Report that scope;
   do not apply SGLang-specific definitions to vLLM. Bound tensor dumps before a short run.
3. Review definitions against executed source and input layouts. Native templates are
   evidence, not automatically correct. See [capture contracts](references/non_fi_capture.md).
   Run `check_definitions()` before GPU collection. If no FlashInfer implementation exists,
   use the supported SGLang module or callable path rather than silently skipping it.
4. Run a fresh process with reviewed definitions and the inventory. Reuse exact requests
   for comparisons. Inspect `report.json`: distinguish format validity, selected-call
   coverage, and numerical replay. A nonempty workload file alone is not complete coverage.
5. Fix mismatched definitions and recapture in a fresh directory. Include both prefill and
   decode where relevant; never reshape inputs silently to hide a layout mismatch.
   Check numerical reference behavior using existing baseline/test tools before claiming
   correctness. Stateful operators require initial-state and updated-state validation.

The example deliberately disables CUDA Graph: Python hooks do not observe replay as normal
calls. Graph-specific capture is separate work and must not be implied by an eager smoke.
Keep raw captures on failure. After successful inspection, generated raw dumps may be removed
while retaining definitions, metadata, reports, and output data. Stop a bounded attempt and
report a concrete blocker rather than retrying indefinitely. Publication is a separate action
and requires the user's instruction; this skill does not authorize sending PRs or messages.
