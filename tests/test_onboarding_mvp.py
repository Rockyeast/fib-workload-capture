"""MVP regressions: partial coverage, layout variants, snapshots, and cleanup."""

import json

import pytest
import torch

from fib_workload_capture import capture, export
from fib_workload_capture.checks import check_definitions, coverage_report
from fib_workload_capture.tracing.sglang_dumper_adapter import adapt_sglang_dumper_workloads


def definition(root, rank=1):
    name = f"identity_r{rank}"
    value = {
        "name": name,
        "op_type": "test",
        "axes": {"n": {"type": "var"}, **({"m": {"type": "var"}} if rank == 2 else {})},
        "inputs": {"x": {"shape": ["n"] if rank == 1 else ["n", "m"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["n"] if rank == 1 else ["n", "m"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x\n",
        "tags": ["sglang_module:test.Fake", "sglang_input:x=arg:0"],
    }
    p = root / "definitions/test" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value))
    return p


def evidence(root, definitions):
    for i, p in enumerate(definitions):
        binding = root / "sglang/module_bindings/123" / f"{i}.pt"
        binding.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "definition": p.stem,
                "class_path": "test.Fake",
                "module_path": "model.layers.0.op",
                "argument_positions": {"x": 0},
                "attributes": {},
            },
            binding,
        )
    for step, x in enumerate([torch.ones(4), torch.ones(2, 4)]):
        for index, kind in enumerate(["inputs.0", "output"]):
            path = root / "dumper/capture/TP0_PP0_Rank0_pid123" / f"{step}-{index}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "value": x,
                    "meta": {
                        "name": "non_intrusive__model.layers.0.op." + kind,
                        "rank": 0,
                        "step": step,
                        "dump_index": index,
                    },
                },
                path,
            )


@pytest.mark.parametrize("variants,uncovered", [(1, 1), (2, 0)])
def test_layout_union_coverage(tmp_path, variants, uncovered):
    defs = [definition(tmp_path, rank) for rank in range(1, variants + 1)]
    evidence(tmp_path / "raw", defs)
    report = adapt_sglang_dumper_workloads(
        tmp_path / "raw/dumper", capture_root=tmp_path / "raw/sglang", definition_files=defs
    )
    assert report["summary"]["uncovered_calls"] == uncovered
    assert report["summary"]["covered_calls"] == 2 - uncovered
    result = coverage_report({p.stem: 1 for p in defs}, report, {"summary": {}})
    assert result["ok"] is (uncovered == 0)


def test_context_does_not_overwrite_and_restores_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "old")
    run = tmp_path / "run"
    with pytest.raises(RuntimeError, match="engine failed"):
        with capture(run):
            raise RuntimeError("engine failed")
    import os

    assert os.environ["FLASHINFER_TRACE_DUMP"] == "old"
    assert json.loads((run / "capture.json").read_text())["completed"] is False
    with pytest.raises(FileExistsError):
        with capture(run):
            pass


def test_export_partial_and_complete(tmp_path):
    for variants in (1, 2):
        source = tmp_path / f"source{variants}"
        defs = [definition(source, rank) for rank in range(1, variants + 1)]
        run = tmp_path / f"run{variants}"
        inventory = {
            "modules": [{"class_path": "test.Fake", "module_paths": ["model.layers.0.op"]}]
        }
        with capture(run, definitions=source / "definitions", inventory=inventory):
            evidence(run / "raw", defs)
        result = export(run)
        assert result["structure_ok"]
        assert result["accepted"] is (variants == 2)
        assert result["numerical_replay"] == "not_run"
        # The exporter must never attach stale captures to modified definitions.
        p = run / "definitions/test/identity_r1.json"
        p.write_text(p.read_text() + "\n")
        with pytest.raises(ValueError, match="snapshot"):
            export(run)


def test_bad_native_template_rejected(tmp_path):
    p = definition(tmp_path)
    value = json.loads(p.read_text())
    value["op_type"] = "gqa_paged"
    value["axes"].update(
        {
            "num_qo_heads": {"type": "const", "value": 24},
            "num_kv_heads": {"type": "const", "value": 256},
        }
    )
    target = tmp_path / "definitions/gqa_paged" / p.name
    target.parent.mkdir()
    target.write_text(json.dumps(value))
    p.unlink()
    with pytest.raises(ValueError, match="GQA head"):
        check_definitions(tmp_path / "definitions")


def test_callable_capture_in_spawned_worker(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    source = tmp_path / "source"
    p = definition(source)
    value = json.loads(p.read_text())
    value["tags"] = ["sglang_callable:mvp_fixture.identity"]
    p.write_text(json.dumps(value))
    (tmp_path / "mvp_fixture.py").write_text("def identity(x):\n    return x\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run = tmp_path / "run"
    with capture(run, definitions=source / "definitions"):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; from mvp_fixture import identity; identity(torch.ones(4))",
            ],
            check=True,
            timeout=60,
        )
    result = export(run)
    assert result["accepted"]
    assert result["counts"]["identity_r1"] == 1
    assert result["workers"]["summary"]["capture_calls"] == 1
    assert len(list((run / "raw/sglang/status").glob("*.json"))) >= 2


def test_module_status_survives_worker_without_atexit(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    source = tmp_path / "source"
    p = definition(source)
    value = json.loads(p.read_text())
    value["tags"] = ["sglang_module:mvp_fixture.Fake", "sglang_input:x=arg:0"]
    p.write_text(json.dumps(value))
    (tmp_path / "mvp_fixture.py").write_text(
        "import torch\nclass Fake(torch.nn.Module):\n    def forward(self, x): return x\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run = tmp_path / "run"
    inventory = {"modules": [{"class_path": "mvp_fixture.Fake", "module_paths": ["op"]}]}
    with capture(run, definitions=source / "definitions", inventory=inventory):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import os,torch; from mvp_fixture import Fake; "
                "root=torch.nn.Module(); root.op=Fake(); root.op(torch.ones(4)); os._exit(0)",
            ],
            check=True,
            timeout=60,
        )
    reports = [json.loads(p.read_text()) for p in (run / "raw/sglang/status").glob("*.json")]
    assert sum(sum(r["bindings"].values()) for r in reports) >= 1
    assert not any(r["errors"] for r in reports)


def test_native_backend_isolated_from_sglang(tmp_path, monkeypatch):
    import importlib
    import os
    from contextlib import contextmanager

    api = importlib.import_module("fib_workload_capture.capture")

    def forbidden(*args, **kwargs):
        raise AssertionError("native backend must not install SGLang hooks")

    monkeypatch.setattr(api, "sglang_capture_environment", forbidden)
    source = tmp_path / "source"
    p = definition(source)
    value = json.loads(p.read_text())
    value["tags"] = ["fi_api:flashinfer.identity"]
    p.write_text(json.dumps(value))

    @contextmanager
    def native_probe(directory):
        directory.mkdir()
        (directory / p.name).write_text(json.dumps(value))
        yield

    monkeypatch.setattr(api, "flashinfer_definition_dump", native_probe)
    with capture(tmp_path / "probe", backend="flashinfer"):
        pass
    assert (tmp_path / "probe/definitions/test" / p.name).is_file()
    assert not (tmp_path / "probe/inventory.json").exists()
    with capture(tmp_path / "collect", definitions=source / "definitions", backend="flashinfer"):
        assert os.environ["FLASHINFER_LOGLEVEL"] == "10"
    assert json.loads((tmp_path / "collect/capture.json").read_text())["backend"] == "flashinfer"


def test_native_backend_rejects_foreign_definitions(tmp_path):
    definition(tmp_path / "source")
    with pytest.raises(ValueError, match="only supports fi_api"):
        with capture(
            tmp_path / "run", backend="flashinfer", definitions=tmp_path / "source/definitions"
        ):
            pytest.fail("invalid definition reached caller inference")
    assert not (tmp_path / "run").exists()
    with pytest.raises(ValueError, match="backend"):
        with capture(tmp_path / "run", backend="unknown"):
            pass


def test_native_dump_limits_restore_environment(tmp_path, monkeypatch):
    import os
    from fib_workload_capture.tracing.flashinfer_logging import flashinfer_workload_dump

    p = definition(tmp_path)
    value = json.loads(p.read_text())
    value["tags"] = ["fi_api:flashinfer.identity"]
    p.write_text(json.dumps(value))
    monkeypatch.setenv("FLASHINFER_DUMP_MAX_COUNT", "8")
    monkeypatch.setenv("FLASHINFER_DUMP_MAX_SIZE_GB", "2")
    monkeypatch.setenv("FLASHINFER_LOGLEVEL", "0")
    with flashinfer_workload_dump([p], tmp_path / "dump"):
        assert os.environ["FLASHINFER_DUMP_MAX_COUNT"] == "8"
        assert os.environ["FLASHINFER_DUMP_MAX_SIZE_GB"] == "2"
        assert os.environ["FLASHINFER_LOGLEVEL"] == "10"
    assert os.environ["FLASHINFER_LOGLEVEL"] == "0"


def test_strict_native_paged_inputs(tmp_path):
    import json
    import torch
    from safetensors.torch import save_file, load_file
    from fib_workload_capture.tracing.sanitize import _strict_native_call

    call = tmp_path / "call"
    plan = tmp_path / "plan"
    call.mkdir()
    plan.mkdir()
    owner = {"type": "Wrapper", "repr": "<Wrapper at 0x1>"}
    record = {
        "execution_status": "completed",
        "function_name": "Wrapper.run",
        "function_signature": "(self, q, paged_kv_cache)",
        "input_metadata": {"arg_0": owner},
    }
    plan_record = {
        "execution_status": "completed",
        "function_name": "Wrapper.plan",
        "function_signature": "(self, indptr, indices, last_page_len, sm_scale=None)",
        "input_metadata": {"arg_0": owner, "kwarg_sm_scale": 0.125},
    }
    (call / "metadata.jsonl").write_text(json.dumps(record) + "\n")
    (plan / "metadata.jsonl").write_text(json.dumps(plan_record) + "\n")
    cache = torch.arange(24, dtype=torch.float32).reshape(3, 1, 4, 2)
    save_file(
        {"arg_1": torch.ones(1, 2, 2), "arg_2__0": cache, "arg_2__1": cache.clone()},
        str(call / "inputs.safetensors"),
    )
    save_file(
        {
            "arg_1": torch.tensor([0, 2], dtype=torch.int32),
            "arg_2": torch.tensor([2, 0, 999], dtype=torch.int32),
            "arg_3": torch.tensor([1], dtype=torch.int32),
        },
        str(plan / "inputs.safetensors"),
    )
    axes = {k: {"type": "var"} for k in ["B", "P", "I", "N"]}
    axes.update(
        {k: {"type": "const", "value": v} for k, v in {"H": 2, "K": 1, "S": 4, "D": 2}.items()}
    )
    definition = {
        "op_type": "gqa",
        "axes": axes,
        "inputs": {
            "q": {"shape": ["B", "H", "D"], "dtype": "float32"},
            "k_cache": {"shape": ["P", "K", "S", "D"], "dtype": "float32"},
            "v_cache": {"shape": ["P", "K", "S", "D"], "dtype": "float32"},
            "kv_indptr": {"shape": ["I"], "dtype": "int32"},
            "kv_indices": {"shape": ["N"], "dtype": "int32"},
            "kv_last_page_len": {"shape": ["B"], "dtype": "int32"},
            "sm_scale": {"shape": None, "dtype": "float32"},
        },
    }
    result = _strict_native_call(
        call, definition, "test", tmp_path / "blob" / "workloads", "Wrapper.run", plan
    )
    assert result["workload"]["axes"] == {"B": 1, "P": 3, "I": 2, "N": 2}
    assert result["workload"]["inputs"]["sm_scale"]["value"] == 0.125
    spec = result["workload"]["inputs"]["kv_indices"]
    data = load_file(str(tmp_path / spec["path"]))
    assert data["kv_indices"].tolist() == [2, 0]
    assert torch.equal(data["k_cache"], cache)
    assert data["kv_last_page_len"].tolist() == [1]
    # Wrong wrapper instance, missing structural input, and wrong layouts must fail closed.
    plan_record["input_metadata"]["arg_0"] = {"repr": "<Wrapper at 0x2>"}
    (plan / "metadata.jsonl").write_text(json.dumps(plan_record) + "\n")
    assert _strict_native_call(call, definition, "test", tmp_path, "Wrapper.run", plan) is None
    plan_record["input_metadata"]["arg_0"] = owner
    (plan / "metadata.jsonl").write_text(json.dumps(plan_record) + "\n")
    definition["inputs"]["missing_indices"] = {"shape": ["N"], "dtype": "int32"}
    assert _strict_native_call(call, definition, "test", tmp_path, "Wrapper.run", plan) is None
    del definition["inputs"]["missing_indices"]
    definition["axes"]["S"]["value"] = 8
    assert _strict_native_call(call, definition, "test", tmp_path, "Wrapper.run", plan) is None

    # Interleaved wrappers: A.plan, B.plan, A.run must use A's earlier plan.
    import shutil
    from fib_workload_capture.tracing.sanitize import sanitize_dumps

    definition["axes"]["S"]["value"] = 4
    definition.update(name="test", tags=["fi_api:Wrapper.run"])
    def_file = tmp_path / "definition.json"
    def_file.write_text(json.dumps(definition))
    raw = tmp_path / "interleaved"
    raw.mkdir()
    a_plan = raw / "20260101_000001_000_pid123_Wrapper.plan_call0001"
    b_plan = raw / "20260101_000002_000_pid123_Wrapper.plan_call0002"
    a_run = raw / "20260101_000003_000_pid123_Wrapper.run_call0001"
    shutil.copytree(plan, a_plan)
    shutil.copytree(plan, b_plan)
    shutil.copytree(call, a_run)
    other = dict(plan_record, input_metadata={"arg_0": {"repr": "<Wrapper at 0x2>"}})
    (b_plan / "metadata.jsonl").write_text(json.dumps(other) + "\n")
    result = sanitize_dumps(raw, [def_file], tmp_path / "export", strict=True)
    assert len(result["test"]) == 1
    assert result["test"][0]["workload"]["inputs"]["sm_scale"]["value"] == 0.125
