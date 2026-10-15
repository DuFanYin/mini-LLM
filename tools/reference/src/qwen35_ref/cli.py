"""qwen35-ref: run the NumPy reference and write golden files.

qwen35-ref generate "The capital of France is" --max-new 16
qwen35-ref dump --out ../../weights/golden          # one .safetensors per built-in prompt
"""

import argparse
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from . import safetensors
from .model import Qwen35

REPO = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = REPO / "weights" / "qwen3.5-0.8b"
DEFAULT_GOLDEN = REPO / "weights" / "golden"

# Built-in prompts for golden files: short, deterministic, covering both a
# factual completion and a longer span that exercises the recurrence and cache.
PROMPTS = {
    "capital": "The capital of France is",
    "count": "1, 2, 3, 4, 5,",
    "code": "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
}


def _load(checkpoint: Path) -> tuple[Qwen35, Tokenizer]:
    t0 = time.perf_counter()
    model = Qwen35(checkpoint)
    tok = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    print(f"loaded {checkpoint.name} in {time.perf_counter() - t0:.1f}s")
    return model, tok


def cmd_generate(args: argparse.Namespace) -> None:
    model, tok = _load(args.checkpoint)
    ids = tok.encode(args.prompt).ids
    t0 = time.perf_counter()
    out = model.greedy(ids, args.max_new)
    dt = time.perf_counter() - t0
    print(f"prompt ids {ids}")
    print(f"greedy ids {out}")
    print(repr(args.prompt + tok.decode(out)))
    print(f"{len(out)} tokens in {dt:.1f}s")


def cmd_dump(args: argparse.Namespace) -> None:
    model, tok = _load(args.checkpoint)
    for name, prompt in PROMPTS.items():
        ids = tok.encode(prompt).ids
        trace: dict[str, np.ndarray] = {}
        state = model.new_state()
        model.forward(ids, state, trace)
        greedy = model.greedy([int(np.argmax(trace["logits"][-1]))], args.max_new - 1, state)
        greedy = [int(np.argmax(trace["logits"][-1]))] + greedy
        tensors = {"input_ids": np.asarray(ids, np.int32), "greedy_ids": np.asarray(greedy, np.int32)}
        for key, value in trace.items():
            # Full logits are 1 MB per position; keep the last row only.
            tensors[key] = value[-1:] if key == "logits" else value
        path = args.out / f"{name}.safetensors"
        safetensors.save(path, tensors, {"prompt": prompt, "checkpoint": args.checkpoint.name})
        print(f"{path}: {len(ids)} prompt tokens -> {tok.decode(greedy)!r}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="qwen35-ref", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    sub = ap.add_subparsers(required=True)

    g = sub.add_parser("generate", help="greedy-decode a prompt")
    g.add_argument("prompt")
    g.add_argument("--max-new", type=int, default=16)
    g.set_defaults(func=cmd_generate)

    d = sub.add_parser("dump", help="write golden traces for the built-in prompts")
    d.add_argument("--out", type=Path, default=DEFAULT_GOLDEN)
    d.add_argument("--max-new", type=int, default=8)
    d.set_defaults(func=cmd_dump)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
