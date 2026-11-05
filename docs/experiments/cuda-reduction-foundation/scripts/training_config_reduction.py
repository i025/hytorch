"""Keep preprocessing/prefetch matched; isolate Sum and existing ReLU prototypes."""
METHODS = ('A', 'I', 'S', 'R', 'SR', 'E')


def configuration(method, model, protocol, prefetch=2):
    return dict(method=method, engine=method, workers=2, prefetch_factor=prefetch,
        model=model, open_stream=True, batch=4, dtype='float32', seed=42, lr=.001,
        dropout=0, tf32=False, warmup_steps=10, indices=protocol['throughput'],
        preprocessing=protocol['preprocessing'], no_fingerprints=True,
        reduction='native' if method in ('S','SR') else 'original',
        relu_backward='fused' if method in ('R','SR') else 'original',
        new_cuda_prototype='Sum only; ReLU inherited unchanged',
        cpu_prefetch='Same existing bounded AsyncDataLoader for ALL methods')
