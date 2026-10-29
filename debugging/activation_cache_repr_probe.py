"""Probe cache display and hook registration without loading model weights.

Run from the checkout: uv run python debugging/activation_cache_repr_probe.py
Optional: --device cuda --repeats 5 --keys 32 1024 --elements 1 1000000
In a notebook: %run debugging/activation_cache_repr_probe.py --display

Timings exclude imports, allocation and model execution. Each synthetic cache shares
one tensor across its keys, so key count and tensor size vary independently.
Formatting timings exclude notebook transport/browser rendering. --display emits
one small cache for a manual frontend check; it does not time browser completion.
"""

import argparse
import inspect
import io
import statistics
import subprocess
import sys
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Callable
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import IPython
import torch
from IPython.core.formatters import DisplayFormatter
from IPython.display import display

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint


class ReprTouched(RuntimeError):
    """A guarded value was represented."""


class GuardModel:
    def __repr__(self) -> str:
        raise ReprTouched("model representation reached")


class CountingHook:
    def __init__(self) -> None:
        self.repr_calls = 0

    def __call__(self, tensor: torch.Tensor, *, hook: HookPoint) -> None:
        pass

    def __repr__(self) -> str:
        self.repr_calls += 1
        return "CountingHook()"


def captured_print(value: object) -> str:
    stream = io.StringIO()
    print(value, file=stream)
    return stream.getvalue()


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def verify_source(cls: type, filename: str) -> None:
    actual = Path(inspect.getfile(cls)).resolve()
    expected = (ROOT / "transformer_lens" / filename).resolve()
    if actual != expected:
        raise RuntimeError(f"Wrong source for {cls.__name__}: {actual}; expected {expected}")
    print(f"{cls.__name__}: {actual}")


def check_value_access(format_text: Callable[[object], str]) -> None:
    tensor = torch.zeros(1)
    cache = ActivationCache({"hook_embed": tensor}, GuardModel())
    with patch.object(
        torch.Tensor, "__repr__", side_effect=ReprTouched("tensor repr reached")
    ) as tensor_repr:
        for label, operation in (
            ("repr(cache)", lambda: repr(cache)),
            ("str(cache)", lambda: str(cache)),
            ("print(cache) -> StringIO", lambda: captured_print(cache)),
            ("IPython text/plain(cache)", lambda: format_text(cache)),
        ):
            result = operation()
            tensor_repr.assert_not_called()
            if not result:
                raise RuntimeError(f"No representation returned by {label}")
            print(f"PASS {label}: no tensor/model repr")
        try:
            repr(cache.cache_dict)
        except ReprTouched:
            print("PASS dict control: tensor repr reached")
        else:
            raise RuntimeError("Guard control failed: dictionary did not represent its tensor")


def check_hook_registration() -> None:
    def callback(tensor: torch.Tensor, *, hook: HookPoint, payload: object) -> None:
        pass

    guarded = partial(callback, payload=GuardModel())
    hp = HookPoint()
    try:
        hp.add_hook(guarded)
        print("PASS partial registration: bound payload repr skipped")
        try:
            repr(hp.fwd_hooks)
        except ReprTouched:
            print("OBSERVED handle-list display: bound payload repr reached")
        else:
            print("OBSERVED handle-list display: bound payload repr not reached")
    finally:
        hp.remove_hooks(including_permanent=True)

    for direction in ("fwd", "bwd"):
        counter = CountingHook()
        hp = HookPoint()
        try:
            if direction == "fwd":
                hp.add_hook(counter, dir="fwd")
            else:
                hp.add_hook(counter, dir="bwd")
            print(f"OBSERVED {direction} registration: callable repr calls={counter.repr_calls}")
        finally:
            hp.remove_hooks(dir="both", including_permanent=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=positive_int, nargs="+", default=[32, 1024])
    parser.add_argument("--elements", type=positive_int, nargs="+", default=[1, 1000000])
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--repr-iterations", type=positive_int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--display", action="store_true", help="emit one small cache in IPython")
    args = parser.parse_args()
    if args.display and IPython.get_ipython() is None:
        parser.error("--display requires IPython; use %run in a notebook")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")

    verify_source(ActivationCache, "ActivationCache.py")
    verify_source(HookPoint, "hook_points.py")
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"commit={commit} python={sys.version.split()[0]}")
    print(f"torch={torch.__version__} IPython={IPython.__version__} device={args.device}")
    print("Fresh IPython formatter; existing notebook formatter registrations are not reused.")
    print("PyTorch print options are inherited unchanged from this process.")
    formatter = DisplayFormatter()

    def format_text(value: object) -> str:
        data, _ = formatter.format(value, include=["text/plain"])
        return data.get("text/plain", "")

    check_value_access(format_text)
    check_hook_registration()

    def synchronize() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()
        elif args.device == "mps":
            torch.mps.synchronize()

    def measure(label: str, operation: Callable[[], str], iterations: int = 1) -> tuple[float, str]:
        operation()
        samples = []
        result = ""
        for _ in range(args.repeats):
            synchronize()
            start = perf_counter()
            for _ in range(iterations):
                result = operation()
            synchronize()
            samples.append((perf_counter() - start) * 1000 / iterations)
        median_ms = statistics.median(samples)
        print(f"  {label:30s} {median_ms:10.3f} ms  {len(result):10d} chars")
        return median_ms, result

    print("\nMedian timings; print is captured, and no large representations are emitted.")
    print(f"repr(cache) uses {args.repr_iterations} calls per sample to reduce timer noise.")
    comparisons: dict[int, list[tuple[int, float, str]]] = {}
    for elements in args.elements:
        tensor = torch.ones(elements, device=args.device, dtype=torch.float32)
        for keys in args.keys:
            cache = ActivationCache(
                {f"blocks.{i}.hook_resid_pre": tensor for i in range(keys)}, None
            )
            print(f"keys={keys} elements/tensor={elements} shared_storage_bytes={elements * 4}")
            elapsed, text = measure("repr(cache)", lambda: repr(cache), args.repr_iterations)
            comparisons.setdefault(keys, []).append((elements, elapsed, text))
            measure("str(cache)", lambda: str(cache))
            measure("print(cache) -> StringIO", lambda: captured_print(cache))
            measure("IPython text/plain(cache)", lambda: format_text(cache))
        small_cache = ActivationCache({f"hook_{i}": tensor for i in range(4)}, None)
        print(f"Tensor-output controls: 4 keys, elements/tensor={elements}")
        measure("repr(tensor)", lambda: repr(tensor))
        measure("repr(cache_dict)", lambda: repr(small_cache.cache_dict))
        measure("IPython text/plain(dict)", lambda: format_text(small_cache.cache_dict))
        measure("repr((tensor, cache))", lambda: repr((tensor, small_cache)))

    print("\nFixed-key comparison: only tensor element count changes within each group.")
    print("Ratios are observations, not significance tests; rerun to assess timing noise.")
    for keys, rows in comparisons.items():
        baseline_elements, baseline_ms, baseline_text = rows[0]
        print(f"keys={keys}; baseline elements/tensor={baseline_elements}")
        for elements, elapsed, text in rows:
            ratio = elapsed / baseline_ms if baseline_ms else float("nan")
            print(
                f"  elements={elements:10d} repr={elapsed:.6f} ms "
                f"vs_baseline={ratio:.3f}x identical_repr={text == baseline_text}"
            )
    if len(set(args.elements)) < 2:
        print("Supply at least two distinct --elements values to compare tensor sizes.")

    if args.display:
        print("Manual frontend check: displaying a small wrapper (not timed).")
        display(ActivationCache({"hook_embed": torch.zeros(1)}, None))
    print(
        "\nNo model inference was run. These results do not reproduce the reported notebook hang."
    )


if __name__ == "__main__":
    main()
