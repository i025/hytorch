"""Small reusable loop demonstrating explicit data-load and phase boundaries.

Callbacks keep model/loss/optimizer semantics with the caller. No GPU transfers,
synchronization, scalar extraction or logging are added outside those callbacks.
The caller owns monitor.start/stop and the final training completion boundary.
"""
import time
from hytorch.dataloaders import DataLoader


def monitored_epochs(dataset, batch_size, epochs, *, monitor, transfer, predict,
                     loss_fn, backward, update, retain_loss=lambda x: x,
                     prepare_batch=lambda batch: batch, num_workers=0,
                     steps_per_epoch=0, phase_context=None):
    """Return retained losses and epoch *host submission* times, not GPU latency.

    ``retain_loss`` should detach a scalar array from its graph without copying
    it to CPU. Read the returned values only after a final completion boundary.
    ``phase_context`` optionally supplies diagnostic NVTX contexts; omitted in
    normal training. ``data_prepare`` is an identity operation by default.
    """
    from contextlib import nullcontext
    extra = phase_context or (lambda _: nullcontext())
    losses, epoch_rows = [], []
    step_id = 0
    for epoch in range(1, epochs+1):
        epoch_start = time.perf_counter_ns()
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
        iterator = iter(loader)
        limit = loader.max_iter if not steps_per_epoch else min(loader.max_iter, steps_per_epoch)
        try:
            for _ in range(limit):
                with monitor.step(step_id, epoch=epoch), extra("step"):
                    with monitor.phase("data_load", gpu=False), extra("data_load"):
                        batch = next(iterator)
                    with monitor.phase("data_prepare", gpu=False), extra("data_prepare"):
                        x, target = prepare_batch(batch)
                        count = len(x)
                        monitor.set_step_metadata(batch_size=count, data_bytes=x.nbytes+target.nbytes)
                    with monitor.phase("h2d"), extra("h2d"):
                        x, target = transfer(x), transfer(target)
                    with monitor.phase("forward"), extra("forward"):
                        prediction = predict(x)
                    with monitor.phase("loss"), extra("loss"):
                        loss = loss_fn(prediction, target)
                    with monitor.phase("backward"), extra("backward"):
                        backward(loss)
                    with monitor.phase("optimizer"), extra("optimizer"):
                        update()
                losses.append((epoch, count, retain_loss(loss)))
                step_id += 1
        finally:
            if hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
        epoch_rows.append(dict(epoch=epoch, host_submission_s=(time.perf_counter_ns()-epoch_start)/1e9))
    return losses, epoch_rows
