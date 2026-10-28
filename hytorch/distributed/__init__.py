"""Single-machine single-process, Parameter Server and Ring AllReduce training.

Public integration API: TrainingConfig, run_training, run_benchmark.
Imports are lazy so importing the package never starts child processes.
"""
from .config import TrainingConfig


def run_training(config):
    from .launcher import run_training as launch
    return launch(config)


def run_benchmark(config, cases=None):
    from .benchmark import run_benchmark as benchmark
    return benchmark(config, cases=cases)


__all__ = ["TrainingConfig", "run_training", "run_benchmark"]
