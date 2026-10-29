"""Deterministic checks for reviewed capture definitions; no model orchestration."""

import ast
import hashlib
import json
from pathlib import Path

from flashinfer_bench.data import Definition

CAPTURE_PREFIXES = ("fi_api:", "sglang_module:", "sglang_callable:")
INTERNAL_PREFIXES = ("sglang_module:", "sglang_callable:", "sglang_input:")


def digest(directory):
    result = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*.json")):
        result.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes())
    return result.hexdigest()


def check_definitions(directory):
    """Return parsed definitions or fail before any GPU work. Schema checks are reused."""
    values = {}
    for path in sorted(Path(directory).rglob("*.json")):
        value = json.loads(path.read_text())
        definition = Definition.model_validate(value)
        if path.stem != definition.name or path.parent.name != definition.op_type:
            raise ValueError(f"{path}: expected op_type/name.json")
        if definition.name in values:
            raise ValueError(f"duplicate definition: {definition.name}")
        tags = value.get("tags", [])
        captures = [tag for tag in tags if tag.startswith(CAPTURE_PREFIXES)]
        if len(captures) != 1 or not captures[0].partition(":")[2]:
            raise ValueError(f"{path}: choose one exact capture target")
        if captures[0].startswith("fi_api:") and captures[0].rsplit(".", 1)[-1][:1].isupper():
            raise ValueError(f"{path}: capture a callable, not a wrapper class")
        mapped = set()
        for tag in tags:
            if not tag.startswith("sglang_input:"):
                continue
            name, _, source = tag.removeprefix("sglang_input:").partition("=")
            kind, _, target = source.partition(":")
            if (
                name not in definition.inputs
                or name in mapped
                or kind not in ("arg", "attr")
                or not target
            ):
                raise ValueError(f"{path}: invalid input mapping {tag}")
            if kind == "attr" and not captures[0].startswith("sglang_module:"):
                raise ValueError(f"{path}: attributes require module capture")
            mapped.add(name)
        axes = value["axes"]
        heads = [axes.get(name, {}).get("value") for name in ("num_qo_heads", "num_kv_heads")]
        if definition.op_type.startswith("gqa_") and all(type(n) is int for n in heads):
            q, kv = heads
            if kv <= 0 or q < kv or q % kv:
                raise ValueError(f"{path}: invalid GQA head counts {heads}")
        run = next(
            node
            for node in ast.parse(value["reference"]).body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        arities = {
            len(node.value.elts)
            if isinstance(node.value, (ast.Tuple, ast.List))
            else int(node.value is not None)
            for node in ast.walk(run)
            if isinstance(node, ast.Return)
        }
        if arities != {len(definition.outputs)}:
            raise ValueError(f"{path}: reference output count differs from definition")
        values[definition.name] = (path, value)
    if not values:
        raise ValueError("no definitions")
    return values


def publish_definition(value):
    """Remove capture metadata; source review never implies numerical verification."""
    result = dict(value)
    result["tags"] = [
        tag
        for tag in value.get("tags", [])
        if not tag.startswith(INTERNAL_PREFIXES) and tag != "status:source_reviewed"
    ]
    if "status:source_reviewed" in value.get("tags", []) and not any(
        tag.startswith("status:") for tag in result["tags"]
    ):
        result["tags"].append("status:unverified")
    return result


def coverage_report(counts, dumper, workers):
    """Union coverage across layout variants, not the sum of per-variant rejects."""
    missing = [name for name, count in counts.items() if not count]
    summary = dumper.get("summary", {})
    errors = summary.get("errors", 0) + workers.get("summary", {}).get("errors", 0)
    uncovered = summary.get("uncovered_calls", 0)
    return {
        "scope": "reviewed definitions and selected SGLang module instances; not all model kernels",
        "definitions_with_workloads": len(counts) - len(missing),
        "missing_definitions": missing,
        "sglang_uncovered_calls": uncovered,
        "errors": errors,
        "ok": not missing
        and not uncovered
        and not errors
        and not summary.get("missing_bindings", 0),
        "native_call_coverage": "definition presence only; native per-call completeness not audited",
    }
