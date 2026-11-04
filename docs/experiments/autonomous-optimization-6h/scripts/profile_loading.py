"""CPU serial preprocessing call profile; includes no CUDA work."""
import argparse
import cProfile
from pathlib import Path
import pstats
import time

from common import load_protocol, write_csv, write_json
from hytorch.dataloaders import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_protocol(args.prepared)
    profile = cProfile.Profile()
    begin = time.perf_counter()
    profile.enable()
    samples = sum(len(x) for x, _ in DataLoader(dataset, 4, shuffle=False, indices=protocol['throughput']))
    profile.disable()
    wall = time.perf_counter() - begin
    profile.dump_stats(args.output / 'serial-loading.pstats')
    stats = pstats.Stats(profile)
    with (args.output / 'serial-loading.txt').open('x') as stream:
        stats.stream = stream
        stats.sort_stats('cumulative').print_stats()
    rows = []
    for (file, line, function), (primitive, total, exclusive, cumulative, callers) in stats.stats.items():
        rows.append(dict(file=file, line=line, function=function, primitive_calls=primitive,
                         total_calls=total, self_s=exclusive, cumulative_s=cumulative))
    rows.sort(key=lambda r: -r['self_s'])
    write_csv(args.output / 'serial-loading.csv', rows)
    write_json(args.output / 'summary.json', dict(status='COMPLETED', samples=samples, wall_s=wall,
        scope='CPU serial cProfile diagnostic; not formal loader throughput and not GPU timing'))


if __name__ == '__main__':
    main()
