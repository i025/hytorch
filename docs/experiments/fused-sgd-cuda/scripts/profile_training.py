"""Trace-only native SGD ranges around the unchanged training benchmark.

Use --nvtx to retain existing phase ranges. This wrapper does not alter the
optimizer, model, or synchronization and is excluded from throughput timing.
"""
import sys

from bench_update import load_training, runtime_ranges


def main():
    if "--nvtx" not in sys.argv:
        raise SystemExit("Profiling wrapper requires --nvtx; use the ordinary benchmark for timing")
    import cupy as cp
    with runtime_ranges(cp):
        load_training().main()


if __name__ == "__main__":
    main()
