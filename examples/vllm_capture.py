"""Capture vLLM's native FlashInfer calls; engine configuration stays caller-owned."""

import argparse
import json
import logging
from pathlib import Path

from fib_workload_capture import capture, export

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--definitions", type=Path)
    parser.add_argument("--engine-config", type=Path, required=True, help="vLLM LLM kwargs JSON")
    parser.add_argument(
        "--requests", type=Path, required=True, help="Exact token IDs and sampling params"
    )
    parser.add_argument("--max-workloads", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    options = json.loads(args.engine_config.read_text())
    requests = json.loads(args.requests.read_text())
    if not options.get("enforce_eager"):
        parser.error("capture example requires enforce_eager=true")
    with capture(
        args.run,
        backend="flashinfer",
        definitions=args.definitions,
        metadata={
            "engine": "vllm",
            "model": args.model,
            "engine_kwargs": options,
            "requests": requests,
        },
    ) as run:
        # Import after logger setup; spawned vLLM workers inherit the environment.
        from vllm import LLM, SamplingParams

        llm = LLM(model=args.model, **options)
        generations = []
        try:
            for request in requests:
                params = request["sampling_params"]
                params = (
                    [SamplingParams(**item) for item in params]
                    if isinstance(params, list)
                    else SamplingParams(**params)
                )
                outputs = llm.generate(
                    [{"prompt_token_ids": ids} for ids in request["input_ids"]],
                    sampling_params=params,
                    use_tqdm=False,
                )
                generations.append(
                    [[list(item.token_ids) for item in out.outputs] for out in outputs]
                )
            (run / "generated_tokens.json").write_text(json.dumps(generations) + "\n")
        finally:
            llm.llm_engine.engine_core.shutdown()
    if args.definitions:
        report = export(run, max_workloads=args.max_workloads)
        logger.info(
            json.dumps({key: report[key] for key in ("accepted", "counts", "coverage")}, indent=2)
        )
        if not report["accepted"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
