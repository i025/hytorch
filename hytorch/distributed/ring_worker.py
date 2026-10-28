"""Decentralized training: ring gradient sum followed by a local optimizer."""
import hashlib
import math
import time
import traceback
from pathlib import Path

import numpy as np
from hytorch import cuda
from hytorch.core import Variable
from hytorch.recorder import MetricRecorder

from .config import TrainingConfig
from .monitor import EventWriter, write_json
from .ring import RingCollective
from .task import epoch_batches, export_parameters, make_optimizer, synchronize
from .worker import accelerator_info, add_numbers, compute_batch, evaluate, make_state, warmup


def parameter_digest(params):
    digest = hashlib.sha256()
    for name, value in export_parameters(params).items():
        digest.update(name.encode('utf-8'))
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def ring_worker_main(config_dict, manifest, run_dir, token, rank, pipe, running_event, stop_event):
    config = TrainingConfig.from_dict(config_dict)
    path = Path(run_dir)
    events = EventWriter(path / 'workers' / f'worker-{rank}.jsonl', path.name, 'ring', rank)
    ring = None
    try:
        events.log('starting', device=config.device)
        ring = RingCollective(config, rank, token, stop_event)
        pipe.send({'ok': True, 'address': ring.address})
        if not pipe.poll(config.startup_timeout_s):
            raise TimeoutError('Launcher did not provide ring addresses')
        addresses = pipe.recv()
        pipe.close()
        ring.connect(addresses)
        task, model, params, xp, dataset = make_state(config, manifest)
        optimizer = make_optimizer(config, model)
        validation_data = task.load_split(manifest, 'val') if rank == 0 else None
        events.log('device_ready', **accelerator_info(xp))
        warm_statistics = warmup(task, model, dataset, xp, config, rank)
        metric_names = sorted(warm_statistics)
        layout = [[name, list(p.data.shape), p.data.dtype.str] for name, p in params.items()]
        initialization, _ = ring.allgather_metadata({'layout': layout, 'digest': parameter_digest(params),
                                                   'metrics': metric_names}, 'initialization')
        if any(value != initialization[0] for value in initialization):
            raise ValueError('Ring initialization/model/statistics differ across ranks')
        dtypes = {p.data.dtype for p in params.values()}
        if len(dtypes) != 1:
            raise ValueError('Ring adapter currently requires a uniform parameter dtype')
        dtype = next(iter(dtypes))
        sizes = [p.data.size for p in params.values()]
        offsets = np.cumsum([0] + sizes)
        gradient_bytes = int(offsets[-1]) * dtype.itemsize
        events.log('running', version=0, left=(rank-1) % config.num_workers,
                   right=(rank+1) % config.num_workers, gradient_tensor_bytes=gradient_bytes)
        running_event.set()
        recorder = MetricRecorder(run_dir=run_dir) if rank == 0 else None
        epochs, version, best = [], 0, float('inf')
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.perf_counter()
            epoch_timings, epoch_statistics = {}, {}
            for batch_index, indices in enumerate(epoch_batches(len(dataset), config.global_batch_size, config.seed, epoch), 1):
                step_start = time.perf_counter()
                shard = np.array_split(indices, config.num_workers)[rank]
                statistics, timing = compute_batch(task, model, dataset, shard, xp)
                copy_start = time.perf_counter()
                gradient = np.zeros(int(offsets[-1]), dtype=dtype)
                if len(shard):
                    for i, param in enumerate(params.values()):
                        gradient[offsets[i]:offsets[i+1]] = cuda.as_numpy(param.grad.data).reshape(-1)
                    gradient *= len(shard) / len(indices)
                synchronize(xp)
                timing['gradient_copy_s'] = time.perf_counter() - copy_start
                stats_array = np.array([statistics.get(name, 0) for name in metric_names], dtype=np.float64)
                reduced, collective = ring.allreduce({'gradient': gradient, 'statistics': stats_array}, f'gradient-{version}')
                timing.update(collective)
                global_statistics = dict(zip(metric_names, reduced['statistics'].tolist()))
                if global_statistics['sample_count'] != len(indices):
                    raise ValueError('Ring global sample count mismatch')
                global_statistics['sample_count'] = int(global_statistics['sample_count'])
                copy_start = time.perf_counter()
                model.cleargrads()
                for i, param in enumerate(params.values()):
                    param.grad = Variable(xp.asarray(reduced['gradient'][offsets[i]:offsets[i+1]].reshape(param.data.shape)))
                synchronize(xp)
                timing['reduced_gradient_copy_s'] = time.perf_counter() - copy_start
                update_start = time.perf_counter()
                optimizer.update()
                synchronize(xp)
                timing['optimizer_s'] = time.perf_counter() - update_start
                version += 1
                timing['compute_s'] = timing['forward_s'] + timing['backward_s']
                timing['step_s'] = time.perf_counter() - step_start
                add_numbers(epoch_timings, timing)
                add_numbers(epoch_statistics, statistics)
                events.log('batch', epoch=epoch, batch=batch_index, step=version,
                           sample_count=len(shard), **timing, **task.quality(statistics))
                if recorder:
                    recorder.log_batch(version, epoch, schema_version=1, role='ring',
                                       samples=len(indices), **task.quality(global_statistics))
                if rank == 0 and (batch_index == 1 or batch_index % 100 == 0):
                    print(f'[Ring {config.num_workers}W] epoch {epoch}/{config.epochs} '
                          f'step {batch_index}/{math.ceil(len(dataset)/config.global_batch_size)} '
                          f'mse={task.quality(global_statistics)["mse"]:.6f} step_s={timing["step_s"]:.3f}', flush=True)
            training_s = time.perf_counter() - epoch_start
            validation, evaluation_s = evaluate(task, model, validation_data, xp, config.eval_batch_size) if rank == 0 else ({}, 0.0)
            digest = parameter_digest(params)
            epoch_reports, _ = ring.allgather_metadata({'rank': rank, 'train_time_s': training_s,
                'statistics': epoch_statistics, 'timings': epoch_timings, 'digest': digest,
                'validation': validation, 'evaluation_time_s': evaluation_s}, f'epoch-{epoch}')
            if len({value['digest'] for value in epoch_reports}) != 1:
                raise ValueError('Ring replicas diverged after optimizer updates')
            totals, times = {}, {}
            for report in epoch_reports:
                add_numbers(totals, report['statistics'])
                add_numbers(times, report['timings'])
            if totals['sample_count'] != manifest['train_size']:
                raise ValueError('Ring epoch samples were duplicated or dropped')
            if rank == 0:
                train_s = max(value['train_time_s'] for value in epoch_reports)
                summary = dict(epoch=epoch, train_time_s=train_s, epoch_wall_s=time.perf_counter()-epoch_start,
                    samples=totals['sample_count'], samples_per_sec=totals['sample_count']/train_s,
                    aggregation_s=times['aggregation_s']/config.num_workers,
                    optimizer_s=times['optimizer_s']/config.num_workers,
                    worker_time_sums=times, train=task.quality(totals), validation=validation,
                    evaluation_time_s=evaluation_s, replica_digests=[r['digest'] for r in epoch_reports])
                recorder.log_epoch(epoch, schema_version=1, role='ring', epoch_time=train_s,
                                   samples_per_sec=summary['samples_per_sec'], **summary['train'],
                                   **{'val_'+key: value for key, value in validation.items()})
                checkpoint_start = time.perf_counter()
                weights = export_parameters(params)
                if any(not np.isfinite(value).all() for value in weights.values()):
                    raise FloatingPointError('Non-finite ring checkpoint')
                np.savez_compressed(path / 'checkpoints' / 'last.npz', **weights)
                if validation['mse'] < best:
                    best = validation['mse']
                    np.savez_compressed(path / 'checkpoints' / 'best.npz', **weights)
                summary['checkpoint_time_s'] = time.perf_counter() - checkpoint_start
                epochs.append(summary)
                write_json(path / 'training_result.json', dict(schema_version=1, epochs=epochs,
                    final_version=version, train_time_s=sum(e['train_time_s'] for e in epochs),
                    validation=validation, checkpoint_kind='model_weights_only',
                    gradient_tensor_bytes=gradient_bytes, optimizer_location=config.device,
                    topology='neighbor_only_ring', replica_consistency='sha256_equal_each_epoch'))
                print(f'[Ring] epoch {epoch} complete: train_s={train_s:.2f} val_mae={validation["mae"]:.6f}', flush=True)
            events.log('epoch', epoch=epoch, train_elapsed_s=training_s, statistics=epoch_statistics,
                       timings=epoch_timings, validation=epoch_reports[0]['validation'], parameter_digest=digest)
            ring.allgather_metadata({'epoch': epoch}, f'checkpoint-saved-{epoch}')
        events.log('completed', version=version, parameter_digest=digest)
    except BaseException as exc:
        events.log('failed', error=str(exc), traceback=traceback.format_exc())
        try:
            pipe.send({'ok': False, 'error': str(exc)})
        except (OSError, EOFError):
            pass
        raise
    finally:
        pipe.close()
        if ring is not None:
            ring.close()
