"""Separate CPU diagnostics; instrumentation numbers are not formal throughput."""
import argparse
import cProfile
from io import BytesIO
from pathlib import Path
import pstats
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from pipeline_common import configurations, load_data, write_csv, write_json
import hytorch.dataloaders as current
from donkeycar_hy.udacity_dataset import preprocess_udacity


def staged(paths, output):
    rows = []
    for index, path in enumerate(paths):
        begin = last = time.perf_counter()
        def mark(name):
            nonlocal last
            now = time.perf_counter()
            values[name] = (now-last)*1000
            last = now
        values = dict(sample=index, path=str(path))
        raw = path.read_bytes()
        mark('file_read_ms')
        with Image.open(BytesIO(raw)) as image:
            image.load()
            mark('jpeg_decode_ms')
            image = image.convert('RGB')
            mark('rgb_ms')
            image = image.crop((0, 70, image.width, image.height-25))
            mark('crop_ms')
            image = image.resize((200, 66), Image.BILINEAR)
            mark('resize_ms')
            value = np.asarray(image, dtype=np.float32)/np.float32(127.5)-np.float32(1)
            value = np.ascontiguousarray(value.transpose(2, 0, 1))
            mark('normalize_chw_ms')
        values['total_ms'] = (time.perf_counter()-begin)*1000
        np.testing.assert_array_equal(value, preprocess_udacity(path))
        rows.append(values)
    write_csv(output / 'preprocess-stages.csv', rows)


def diagnostic(module, dataset, indices, workers, prefetch):
    queue_rows, array_rows, sample_rows = [], [], []
    real_numpy = np
    class TrackedQueue(queue.Queue):
        def put(self, item, *args, **kwargs):
            start = time.perf_counter()
            super().put(item, *args, **kwargs)
            queue_rows.append(dict(queue='result' if self.maxsize else 'index', operation='put',
                ms=(time.perf_counter()-start)*1000, depth=self.qsize(), thread=threading.current_thread().name))
        def get(self, *args, **kwargs):
            start = time.perf_counter()
            value = super().get(*args, **kwargs)
            queue_rows.append(dict(queue='result' if self.maxsize else 'index', operation='get',
                ms=(time.perf_counter()-start)*1000, depth=self.qsize(), thread=threading.current_thread().name))
            return value
    class NumpyProxy:
        def __getattr__(self, name):
            return getattr(real_numpy, name)
        def array(self, *args, **kwargs):
            start = time.perf_counter()
            value = real_numpy.array(*args, **kwargs)
            array_rows.append(dict(ms=(time.perf_counter()-start)*1000, bytes=value.nbytes,
                                   thread=threading.current_thread().name))
            return value
    class TimedDataset:
        def __len__(self):
            return len(dataset)
        def __getitem__(self, index):
            start = time.perf_counter()
            value = dataset[index]
            sample_rows.append(dict(index=int(index), ms=(time.perf_counter()-start)*1000,
                                    thread=threading.current_thread().name))
            return value
    old_queue, old_np = module.queue, module.np
    module.queue = SimpleNamespace(Queue=TrackedQueue, Empty=queue.Empty, Full=queue.Full)
    module.np = NumpyProxy()
    try:
        loader = module.DataLoader(TimedDataset(), 4, shuffle=False, indices=indices,
                                   num_workers=workers, prefetch_factor=prefetch)
        begin = time.perf_counter()
        iterator = iter(loader)
        startup = time.perf_counter()-begin
        threads = list(loader._impl._workers) if workers else []
        pending_peak = 0
        for _ in range(loader.max_iter):
            x, t = next(iterator)
            if workers:
                pending_peak = max(pending_peak, len(loader._impl._pending))
        assert next(iterator, None) is None
        elapsed = time.perf_counter()-begin
        row = dict(workers=workers, prefetch_factor=prefetch, samples=len(indices), wall_s=elapsed,
            startup_ms=startup*1000, sample_work_ms=sum(s['ms'] for s in sample_rows),
            batch_assembly_ms=sum(a['ms'] for a in array_rows),
            assembled_bytes=sum(a['bytes'] for a in array_rows),
            result_get_ms=sum(q['ms'] for q in queue_rows if q['queue']=='result' and q['operation']=='get'),
            result_put_ms=sum(q['ms'] for q in queue_rows if q['queue']=='result' and q['operation']=='put'),
            result_queue_peak=max([q['depth'] for q in queue_rows if q['queue']=='result'] or [0]),
            pending_observed_peak=pending_peak, workers_alive_after=sum(w.is_alive() for w in threads),
            scope='Instrumented CPU diagnostic; worker durations overlap; queue times include locks/wait/scheduling')
    finally:
        module.queue, module.np = old_queue, old_np
    return row, queue_rows, array_rows, sample_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--open-stream', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_data(args.prepared, 'stream' if args.open_stream else 'repaired')
    indices = protocol['throughput']
    staged([dataset.paths[i] for i in indices], args.output)
    profile = cProfile.Profile()
    profile.enable()
    for _ in current.DataLoader(dataset, 4, shuffle=False, indices=indices):
        pass
    profile.disable()
    profile.dump_stats(args.output/'serial.pstats')
    stats = pstats.Stats(profile)
    with (args.output/'serial-profile.txt').open('w') as stream:
        stats.stream = stream
        stats.sort_stats('cumulative').print_stats()
    functions = [dict(file=file, line=line, function=fn, calls=count, self_s=own, cumulative_s=cum)
                 for (file,line,fn),(primitive,count,own,cum,callers) in stats.stats.items()]
    write_csv(args.output/'serial-profile.csv', sorted(functions, key=lambda r:-r['self_s']))
    summaries = []
    for workers, prefetch in configurations():
        row, queues, arrays, samples = diagnostic(current, dataset, indices, workers, prefetch)
        summaries.append(row)
        name = f'w{workers}-p{prefetch}'
        for suffix, values in (('queues',queues), ('arrays',arrays), ('samples',samples)):
            write_csv(args.output/(name+'-'+suffix+'.csv'), values)
    write_csv(args.output/'queue-summary.csv', summaries)
    write_json(args.output/'status.json', dict(status='COMPLETED', gpu='NOT_RUN',
        stage_scope='Explicit file read into bytes then PIL decode, hot cache; original lazy file path profiled separately',
        diagnostic_only=True))
    print('CPU stage/profile/queue diagnostics completed', flush=True)


if __name__ == '__main__':
    main()
