"""One-engine example. Choose GPU/TP/backends externally; no scheduler or agent manager."""

import argparse
import json
import logging
from pathlib import Path

from fib_workload_capture import capture, export

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--definitions", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument(
        "--engine-config", type=Path, required=True, help="SGLang Engine kwargs JSON"
    )
    parser.add_argument(
        "--requests", type=Path, required=True, help="Exact token IDs and sampling params JSON"
    )
    parser.add_argument("--max-workloads", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    options = json.loads(args.engine_config.read_text())
    requests = json.loads(args.requests.read_text())
    # Python hooks cannot observe CUDA Graph replay. Capture with graphs disabled.
    if not options.get("disable_cuda_graph"):
        parser.error("capture example requires disable_cuda_graph=true")
    with capture(
        args.run,
        definitions=args.definitions,
        inventory=args.inventory,
        metadata={"model": args.model, "engine_kwargs": options, "requests": requests},
    ) as run:
        # Import after capture setup so spawned workers inherit native logger settings.
        import sglang as sgl

        engine = sgl.Engine(model_path=args.model, **options)
        try:
            for request in requests:
                engine.generate(
                    input_ids=request["input_ids"], sampling_params=request["sampling_params"]
                )
        finally:
            engine.shutdown()
    if args.definitions:
        report = export(run, max_workloads=args.max_workloads)
        logger.info(
            json.dumps(
                {
                    "accepted": report["accepted"],
                    "counts": report["counts"],
                    "coverage": report["coverage"],
                },
                indent=2,
            )
        )
        if not report["accepted"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
