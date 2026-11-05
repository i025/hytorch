"""Extract measured CPU call counts/self times without summing nested cumulative times."""
import argparse
import csv
from pathlib import Path

from common import write_csv, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(a.input.open()))
    selected = []
    for row in rows:
        file, name = row['file'], row['function']
        core = file.endswith('/hytorch/core.py')
        if not ((core and (name in ('__init__', '__call__', 'as_array', 'as_variable', 'add_func', '<lambda>') or
                         (name == 'backward' and int(row['line']) < 180))) or
                name in ('isscalar', 'get_array_module', "<method 'sort' of 'list' objects>")):
            continue
        steps = int(row['iterations'])
        selected.append(dict(workload=row['workload'], function=name, file=file, line=row['line'],
                             iterations=steps, total_calls=int(row['calls']), calls_per_step=int(row['calls'])/steps,
                             self_s=float(row['self_s']), self_us_per_step=float(row['self_s'])*1e6/steps,
                             cumulative_s=float(row['cumulative_s']),
                             scope='CPU cProfile; self time can include allocations/GC/C calls, not GPU host time'))
    write_csv(a.output / 'calls.csv', selected)
    write_json(a.output / 'summary.json', dict(status='ANALYZED', rows=len(selected), gpu='NOT_RUN',
               source=str(a.input.resolve()),
               limitation='Call counts are from CPU graphs. Profiled self times are not uninstrumented timings or Python-exclusive costs.'))


if __name__ == '__main__':
    main()
