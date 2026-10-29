"""Merge worker evidence without choosing deployment or serving settings."""

import json
import re
import sys
from pathlib import Path


def summarize_module_inventory(capture_root):
    modules = {}
    files = sorted((Path(capture_root) / "inventory").glob("*.jsonl"))
    for path in files:
        for line in path.read_text().splitlines():
            item = json.loads(line)
            name = item["class_path"]
            if name not in modules:
                modules[name] = item
            else:
                current = modules[name]
                current["module_paths"] = sorted(
                    set(current["module_paths"]) | set(item["module_paths"])
                )
    return {
        "summary": {"modules": len(modules), "worker_files": len(files)},
        "modules": [modules[name] for name in sorted(modules)],
    }


def summarize_sglang_capture_status(capture_root):
    captures, bindings, errors = {}, {}, []
    files = sorted((Path(capture_root) / "status").glob("*.json"))
    for path in files:
        item = json.loads(path.read_text())
        for key, total in [("captures", captures), ("bindings", bindings)]:
            for name, count in item.get(key, {}).items():
                total[name] = total.get(name, 0) + count
        errors.extend(item.get("errors", []))
    return {
        "summary": {
            "processes": len(files),
            "capture_calls": sum(captures.values()),
            "module_bindings": sum(bindings.values()),
            "errors": len(errors),
        },
        "captures": captures,
        "bindings": bindings,
        "errors": errors[:50],
    }


def _layer_index_from_path(path):
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", path)
    return int(match.group(1)) if match else None


def _module_path_order(path):
    index = _layer_index_from_path(path)
    return (index if index is not None else sys.maxsize, path)
