"""Collect free trajectories, then replay one identical state/batch on every backend."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from common import Engine, ROOT, batch, compare, load_protocol, result, write_csv, write_json
from hytorch.backends.cuda import dispatch_stats

CHECKPOINTS = (1, 2, 32, 128, 256, 512, 800, 862, 863, 958, 959, 1023, 1024,
               1400, 1421, 1422, 1583, 1600)
REPLAYS = (1, 256, 512, 800, 863, 959, 1024, 1422, 1583, 1600)


def training_batches(protocol, cache, dataset, steps):
    count = 0
    for epoch, indices in enumerate(protocol['epoch_order'], 1):
        for start in range(0, len(indices), 4):
            rows = indices[start:start + 4]
            x, t = ((cache['x'][start:start + 4], cache['t'][start:start + 4])
                    if epoch == 1 else batch(dataset, rows))
            count += 1
            yield count, epoch, rows, x, t
            if count == steps:
                return


def warmup(engine, prepared):
    data = np.load(prepared / 'throughput-inputs.npz')
    x, t = engine.convert(data['x'][:4]), engine.convert(data['t'][:4])
    for _ in range(10):
        loss = engine.loss(x, t)
        engine.backward(loss)
        engine.update()
    engine.sync()
    engine.restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--no-fingerprints', action='store_true', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol, dataset = load_protocol(args.prepared)
    with np.load(args.cache) as archive:
        cache = {key: archive[key] for key in archive.files}
    started = datetime.now(timezone.utc).isoformat()
    trajectories, checkpoint_errors, replay_errors, historical = [], [], [], []
    checkpoints = [s for s in CHECKPOINTS if s <= args.steps]
    engines = {m: Engine(m) for m in 'AFIE'}
    cp, torch = engines['I'].cp, engines['E'].torch
    properties = cp.cuda.runtime.getDeviceProperties(0)
    write_json(args.output / 'environment.json', dict(logical_gpu=0, cupy=cp.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion(),
        compute_capability=cp.cuda.Device(0).compute_capability,
        pci_bus_id=cp.cuda.Device(0).pci_bus_id, gpu_name=str(properties['name']),
        total_memory_bytes=properties['totalGlobalMem'], torch=torch.__version__,
        torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        torch_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        torch_cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_deterministic=torch.backends.cudnn.deterministic))
    canonical = engines['A'].initial
    old_initial = np.load(ROOT / 'runs/udacity_real_window_02/train-I/initial_weights.npz')
    for key in canonical:
        np.testing.assert_array_equal(canonical[key], old_initial[key])
    for engine in engines.values():
        for key in canonical:
            np.testing.assert_array_equal(engine.initial[key], canonical[key])
    write_json(args.output / 'config.json', dict(steps=args.steps, checkpoints=checkpoints,
        replay_sources=['F', 'I'], replay_steps=[s for s in REPLAYS if s <= args.steps],
        optimizer=dict(name='SGD', lr=.001, momentum=0, weight_decay=0, hidden_state=None),
        original_protocol=str((args.prepared / 'protocol.json').resolve()),
        tf32=False, dropout=0, no_fingerprints=True))
    # Repeat I independently from initial state to check determinism.
    for run in ('A', 'F', 'I', 'E', 'I-repeat'):
        method = run[0]
        engine = engines[method]
        engine.load_state(canonical)
        target = args.output / run
        target.mkdir()
        rows = []
        begin = time.perf_counter()
        with engine.context():
            warmup(engine, args.prepared)
            dispatch_stats(reset=True)
            for step, epoch, indices, x, t in training_batches(protocol, cache, dataset, args.steps):
                if step in checkpoints:
                    np.savez(target / f'before-{step}.npz', **engine.state())
                    np.savez(target / f'batch-{step}.npz', x=x, t=t, indices=np.array(indices))
                    values = result(engine, x, t)
                    value = float(values['loss'])
                    np.savez(target / f'result-{step}.npz', **values)
                else:
                    loss = engine.loss(engine.convert(x), engine.convert(t))
                    engine.backward(loss)
                    engine.update()
                    value = engine.value(loss)
                assert np.isfinite(value), f'Nonfinite loss in {run} at {step}'
                rows.append(dict(run=run, method=method, step=step, epoch=epoch,
                                 loss=value, indices=json.dumps(indices)))
                if step % 256 == 0:
                    print(run, step, value, flush=True)
            engine.sync()
        write_csv(target / 'losses.csv', rows)
        np.savez(target / 'final.npz', **engine.state())
        write_json(target / 'summary.json', dict(status='COMPLETED', wall_s=time.perf_counter() - begin,
            steps=len(rows), dispatch_counts=dispatch_stats()))
        trajectories.extend(rows)
        prior = ROOT / 'runs/udacity_real_window_02' / ('train-' + method) / 'losses.jsonl'
        old = [json.loads(line)['loss'] for line in prior.read_text().splitlines()][:args.steps]
        historical.append(compare(np.array([r['loss'] for r in rows]), np.array(old), 'loss', run=run,
                                  reference='4H-same-method'))

    write_csv(args.output / 'trajectories.csv', trajectories)
    write_csv(args.output / 'history-reproduction.csv', historical)
    for step in checkpoints:
        values = {m: dict(np.load(args.output / m / f'result-{step}.npz')) for m in ('A', 'F', 'I', 'E', 'I-repeat')}
        for method, reference in (('A', 'E'), ('F', 'E'), ('I', 'E'), ('F', 'I'), ('I-repeat', 'I')):
            for name, value in values[method].items():
                checkpoint_errors.append(compare(value, values[reference][name], name,
                    step=step, method=method, reference=reference, scope='free-trajectory'))
    write_csv(args.output / 'checkpoint-errors.csv', checkpoint_errors)

    for source in ('F', 'I'):
        for step in REPLAYS:
            if step > args.steps:
                continue
            state = dict(np.load(args.output / source / f'before-{step}.npz'))
            data = np.load(args.output / source / f'batch-{step}.npz')
            target = args.output / f'replay-{source}-{step}'
            target.mkdir()
            outputs = {}
            for method, engine in engines.items():
                engine.load_state(state)
                with engine.context():
                    outputs[method] = result(engine, data['x'], data['t'])
                np.savez(target / (method + '.npz'), **outputs[method])
            for method, reference in (('A', 'E'), ('F', 'E'), ('I', 'E'), ('F', 'I')):
                for name, value in outputs[method].items():
                    replay_errors.append(compare(value, outputs[reference][name], name,
                        source=source, step=step, method=method, reference=reference,
                        scope='same-state-single-step'))
            print('REPLAY', source, step, 'complete', flush=True)
    write_csv(args.output / 'replay-errors.csv', replay_errors)
    # Isolate optimizer arithmetic: same actual initial parameters and gradients.
    from hytorch.backends.cuda.runtime import multi_tensor_sgd
    cp, torch = engines['I'].cp, engines['E'].torch
    gradient_source = np.load(args.output / 'replay-F-1/I.npz')
    original, fused, gradients, torch_parameters = [], [], [], []
    keys = list(canonical)
    for key in keys:
        p, g = cp.asarray(canonical[key]), cp.asarray(gradient_source['grad/' + key])
        original.append(p - cp.float32(.001) * g)
        fused.append(p.copy())
        gradients.append(g)
        parameter = torch.nn.Parameter(torch.tensor(canonical[key], device='cuda'))
        parameter.grad = torch.tensor(gradient_source['grad/' + key], device='cuda')
        torch_parameters.append(parameter)
    multi_tensor_sgd(fused, gradients, .001)
    torch.optim.SGD(torch_parameters, lr=.001, foreach=False).step()
    optimizer_errors = []
    for key, old, new, tp in zip(keys, original, fused, torch_parameters):
        old, new, target = cp.asnumpy(old), cp.asnumpy(new), tp.detach().cpu().numpy()
        np.savez(args.output / ('sgd-' + key.replace('/', '-') + '.npz'), original=old, fused=new, torch=target)
        optimizer_errors.append(compare(new, old, 'weight/' + key, method='fused', reference='cupy-separate'))
        optimizer_errors.append(compare(target, old, 'weight/' + key, method='torch', reference='cupy-separate'))
    write_csv(args.output / 'isolated-sgd.csv', optimizer_errors)
    # Preserve isolated dWeight operands for independent CPU FP64 dot checks.
    # Hook only the diagnostic call; computation and backend choice are unchanged.
    import hytorch.functions_conv as FC
    from hytorch.backends.cuda import backend, current_config
    original = FC.Conv2DGradW.forward
    operands = []
    def capture(node, x, gy):
        value = original(node, x, gy)
        with backend(**dict(current_config(), conv_dweight='original')):
            old = original(node, x, gy)
        torch = engines['E'].torch
        tx, tg = torch.tensor(engines['I'].cp.asnumpy(x), device='cuda'), torch.tensor(engines['I'].cp.asnumpy(gy), device='cuda')
        tw = torch.nn.grad.conv2d_weight(tx, (gy.shape[1], x.shape[1], *node.kernel_size),
                                         tg, stride=node.stride, padding=node.pad)
        operands.append(dict(x=engines['I'].cp.asnumpy(x), gy=engines['I'].cp.asnumpy(gy),
                             dw=engines['I'].cp.asnumpy(value), original=engines['I'].cp.asnumpy(old),
                             torch=tw.detach().cpu().numpy(), kernel=np.array(node.kernel_size),
                             stride=np.array(node.stride), pad=np.array(node.pad)))
        return value
    for step in (1, 863, 1422):
        if step > args.steps:
            continue
        engine = engines['I']
        engine.load_state(dict(np.load(args.output / 'F' / f'before-{step}.npz')))
        data = np.load(args.output / 'F' / f'batch-{step}.npz')
        operands.clear()
        FC.Conv2DGradW.forward = capture
        try:
            with engine.context():
                result(engine, data['x'], data['t'])
        finally:
            FC.Conv2DGradW.forward = original
        for node, values in enumerate(operands):
            np.savez(args.output / f'dweight-operands-{step}-{node}.npz', **values)
    deterministic = [r for r in checkpoint_errors if r['method'] == 'I-repeat']
    repeat_losses = {run: np.array([r['loss'] for r in trajectories if r['run'] == run]) for run in ('I', 'I-repeat')}
    summary = dict(status='COMPLETED', started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(),
        free_trajectory_comparisons=len(checkpoint_errors), replay_comparisons=len(replay_errors),
        replay_failed_groups=sum(r['mismatches'] > 0 or not r['finite'] for r in replay_errors),
        replay_failed_elements=sum(r['mismatches'] for r in replay_errors),
        replay_max_abs=max(r['max_abs'] for r in replay_errors),
        repeat_bitwise_identical=all(r['bitwise_equal'] for r in deterministic),
        repeat_all_losses_equal=bool(np.array_equal(repeat_losses['I'], repeat_losses['I-repeat'])),
        repeat_loss_max_abs=float(abs(repeat_losses['I'] - repeat_losses['I-repeat']).max()),
        repeat_compared_groups=len(deterministic),
        evidence_scope='Sampled common states; not a proof for every step or every possible input')
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
