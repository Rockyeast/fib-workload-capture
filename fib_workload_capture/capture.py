"""Capture around caller-owned inference, then export using existing tracing tools."""

import json
import os
import shutil
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path

from flashinfer_bench.data.validate import validate_dataset
from fib_workload_capture.tracing.flashinfer_logging import (
    flashinfer_definition_dump,
    flashinfer_workload_dump,
)
from fib_workload_capture.tracing.sanitize import sanitize_dumps
from fib_workload_capture.tracing.sglang_dumper_adapter import (
    adapt_sglang_dumper_workloads,
    build_sglang_dumper_workload_filter,
    merge_sglang_workload_shards,
)
from fib_workload_capture.tracing.sglang_inventory import (
    summarize_module_inventory,
    summarize_sglang_capture_status,
)
from fib_workload_capture.tracing.sglang_worker_capture import sglang_capture_environment
from .checks import check_definitions, coverage_report, digest, publish_definition


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@contextmanager
def dumper_environment(directory, expression):
    updates = {
        "DUMPER_ENABLE": "1",
        "DUMPER_DIR": str(directory),
        "DUMPER_EXP_NAME": "capture",
        "DUMPER_NON_INTRUSIVE_MODE": "all",
        "DUMPER_FILTER": expression,
        "DUMPER_ENABLE_OUTPUT_FILE": "1",
        "DUMPER_ENABLE_OUTPUT_CONSOLE": "0",
        "DUMPER_ENABLE_VALUE": "1",
        "DUMPER_ENABLE_GRAD": "0",
        "DUMPER_ENABLE_MODEL_VALUE": "0",
        "DUMPER_ENABLE_MODEL_GRAD": "0",
    }
    previous = {key: os.environ.get(key) for key in updates}

    def refresh():
        module = sys.modules.get("sglang.srt.debug_utils.dumper")
        if module is not None:
            module.dumper.reset()
            module.dumper.configure(**asdict(module.DumperConfig.from_env()))

    try:
        os.environ.update(updates)
        refresh()
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        refresh()


@contextmanager
def capture(run, *, definitions=None, inventory=None, metadata=None, backend="sglang"):
    """Instrument a fresh run directory; caller owns model startup, requests, and shutdown.

    With no definitions, collect native definitions and an executed-module inventory.
    Otherwise freeze reviewed definitions and collect inputs. Call export() afterwards.
    backend="flashinfer" collects native APIs only, for vLLM or other caller-owned engines;
    it does not install SGLang hooks or discover non-FlashInfer modules.
    Initialize and shut down the serving engine INSIDE this context so workers inherit hooks.
    Use a fresh Python process for each capture (native logger settings may be import-time).
    """
    if backend not in {"sglang", "flashinfer"}:
        raise ValueError("capture backend must be sglang or flashinfer")
    if definitions is not None and backend == "flashinfer":
        for _, value in check_definitions(Path(definitions)).values():
            if not any(tag.startswith("fi_api:") for tag in value["tags"]):
                raise ValueError("flashinfer backend only supports fi_api definitions")
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    raw = run / "raw"
    raw.mkdir()
    write_json(run / "metadata.json", metadata or {})
    values = None
    selection = {}
    if definitions is not None:
        values = check_definitions(Path(definitions))
        shutil.copytree(definitions, run / "definitions")
        values = check_definitions(run / "definitions")
        signature = digest(run / "definitions")
        write_json(run / "definition_snapshot.json", {"sha256": signature})
    try:
        with ExitStack() as stack:
            if values is None:
                stack.enter_context(flashinfer_definition_dump(run / "definitions"))
                if backend == "sglang":
                    stack.enter_context(
                        sglang_capture_environment(raw / "sglang", mode="inventory")
                    )
            else:
                fi = [
                    p for p, v in values.values() if any(t.startswith("fi_api:") for t in v["tags"])
                ]
                sg = [
                    p
                    for p, v in values.values()
                    if any(t.startswith(("sglang_module:", "sglang_callable:")) for t in v["tags"])
                ]
                if fi:
                    stack.enter_context(flashinfer_workload_dump(fi, raw / "flashinfer"))
                if sg:
                    if isinstance(inventory, (str, Path)):
                        inventory = json.loads(Path(inventory).read_text())
                    expression, selection = build_sglang_dumper_workload_filter(sg, inventory or {})
                    if expression:
                        stack.enter_context(dumper_environment(raw / "dumper", expression))
                    stack.enter_context(
                        sglang_capture_environment(
                            raw / "sglang", mode="workloads", definitions_dir=run / "definitions"
                        )
                    )
            yield run
        if values is None:
            # Native tracing writes flat JSON; normalize paths without altering semantics.
            for path in (run / "definitions").glob("*.json"):
                value = json.loads(path.read_text())
                target = run / "definitions" / value["op_type"] / path.name
                target.parent.mkdir(exist_ok=True)
                path.rename(target)
            if backend == "sglang":
                write_json(run / "inventory.json", summarize_module_inventory(raw / "sglang"))
        elif digest(run / "definitions") != signature:
            raise ValueError("reviewed definitions changed during capture")
        write_json(
            run / "capture.json",
            {
                "completed": True,
                "backend": backend,
                "mode": "workloads" if values else "inventory",
                "selection": selection,
            },
        )
    except BaseException as exc:
        write_json(
            run / "capture.json", {"completed": False, "error": f"{type(exc).__name__}: {exc}"}
        )
        raise


def export(run, *, max_workloads=20):
    """Export a completed workload capture; preserve raw evidence for diagnosis/re-export."""
    run = Path(run).resolve()
    capture_status = json.loads((run / "capture.json").read_text())
    if not capture_status["completed"] or capture_status["mode"] != "workloads":
        raise ValueError("export requires a completed workload capture")
    if max_workloads < 1:
        raise ValueError("max_workloads must be positive")
    definitions = run / "definitions"
    values = check_definitions(definitions)
    snapshot = json.loads((run / "definition_snapshot.json").read_text())["sha256"]
    if digest(definitions) != snapshot:
        raise ValueError("definitions differ from captured snapshot; capture again")
    dataset = run / "output"
    dataset.mkdir(exist_ok=False)
    fi, sg = [], []
    for path, value in values.values():
        write_json(
            dataset / "definitions" / path.relative_to(definitions), publish_definition(value)
        )
        (fi if any(t.startswith("fi_api:") for t in value["tags"]) else sg).append(path)
    raw = run / "raw"
    counts = {}
    if fi:
        native = sanitize_dumps(
            dump_dir=raw / "flashinfer",
            definition_files=fi,
            flashinfer_trace_dir=dataset,
            replace=True,
            max_new_workloads=max_workloads,
            strict=capture_status.get("backend") == "flashinfer",
        )
        counts.update({p.stem: len(native.get(p.stem, [])) for p in fi})
    dumper = {"summary": {}}
    workers = {"summary": {}}
    if sg:
        dumper = adapt_sglang_dumper_workloads(
            raw / "dumper", capture_root=raw / "sglang", definition_files=sg
        )
        counts.update(
            merge_sglang_workload_shards(
                raw / "sglang",
                dataset_dir=dataset,
                definition_files=sg,
                max_new_workloads=max_workloads,
            )
        )
        workers = summarize_sglang_capture_status(raw / "sglang")
    validation = validate_dataset(
        dataset=str(dataset),
        checks=["layout", "definition", "workload"],
        outputs=[],
        disable_gpu=True,
        definitions=list(values),
    )
    structure_ok = not any(item.status == "error" for item in validation.definitions.values())
    coverage = coverage_report(counts, dumper, workers)
    if capture_status.get("backend") == "flashinfer":
        coverage["scope"] = "reviewed FlashInfer APIs only; non-FlashInfer operators not captured"
    report = {
        "counts": counts,
        "workloads": sum(counts.values()),
        "coverage": coverage,
        "dataset_validation": validation.model_dump(mode="json"),
        "structure_ok": structure_ok,
        "accepted": structure_ok and coverage["ok"],
        "numerical_replay": "not_run",
        "sglang_dumper": dumper,
        "workers": workers,
        "definitions_sha256": snapshot,
    }
    write_json(run / "report.json", report)
    return report
