"""Reproduce shutdown with a full result queue, without touching CUDA."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


class Dataset:
    def __len__(self):
        return 100

    def __getitem__(self, index):
        return np.array([index], np.float32), np.array([-index], np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('loader_under_test', args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.AsyncDataLoader(Dataset(), 1, shuffle=False, num_workers=2,
                                    prefetch_factor=1)
    iter(loader)
    workers = list(loader._workers)
    deadline = time.monotonic() + 2
    while not loader._result_queue.full() and time.monotonic() < deadline:
        time.sleep(.001)
    assert loader._result_queue.full()
    time.sleep(.02)
    before = dict(result_queue=loader._result_queue.qsize(),
                  alive=sum(w.is_alive() for w in workers))
    begin = time.perf_counter()
    loader._shutdown_workers()
    row = dict(before=before, shutdown_s=time.perf_counter()-begin,
               alive_after_shutdown=sum(w.is_alive() for w in workers),
               retained_worker_references=len(loader._workers),
               status='PASS' if not any(w.is_alive() for w in workers) else 'FAIL')
    args.output.write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
