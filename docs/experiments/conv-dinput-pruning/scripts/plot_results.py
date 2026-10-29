"""Replot recorded CSV results into a new/empty figures directory."""
import argparse
from pathlib import Path
from analyze import csv_rows, plots

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,required=True)
p.add_argument('--figures',type=Path,required=True)
a = p.parse_args()


def read(name, floats=(), integers=()):
    rows = csv_rows(a.results/(name+'.csv'))
    for row in rows:
        for field in floats:
            row[field] = float(row[field])
        for field in integers:
            row[field] = int(row[field])
    return rows


plots(a.figures,read('training-raw',['samples_per_sec']),
      read('training-summary',['mean','std']),read('training-losses',['loss'],['repeat']),
      read('numerical-errors',['max_abs']),read('speedups',['throughput_gain_pct']),
      read('trace-phases'),read('gradient-components'),read('layout-kernels',['gpu_ms']))
