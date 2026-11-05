"""Framework-independent loader policy for the next, separately authorized GPU run."""
METHODS = {
    'A': ('A', 0),
    'I': ('I', 0),
    'P': ('P', 2),
    'R': ('R', 0),
    'J': ('J', 2),
    'E': ('E', 0),
    'E-prefetch': ('E', 2),
}


def configuration(method, model, protocol, prefetch=2):
    engine, workers = METHODS[method]
    return dict(method=method, engine=engine, workers=workers, prefetch_factor=prefetch,
                model=model, open_stream=True, batch=4, dtype='float32', seed=42, lr=.001, dropout=0,
                tf32=False, warmup_steps=10, indices=protocol['throughput'],
                preprocessing=protocol['preprocessing'], no_fingerprints=True,
                new_cuda_prototype='Native ReLU backward only; inherited from previous branch',
                cpu_prefetch='Existing ordered AsyncDataLoader, shared by both frameworks')
