"""Matched HyTorch/CuPy, HyTorch/CUDA C and PyTorch training experiment."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hytorch
import hytorch.functions as F
import hytorch.layers as L
from hytorch import cuda
from hytorch.backends.cuda import backend, dispatch_stats
from hytorch.backends.cuda.runtime import load_library
from hytorch.dataloaders import DataLoader
from hytorch.optimizers import SGD
from hytorch.profiling import StepProfiler, ResourceMonitor, dump_json, diagnose


class TinyNet(L.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = L.Conv2d(4, 3, pad=1)
        self.fc1 = L.Linear(1)

    def forward(self, x):
        x = F.pooling(F.relu(self.conv1(x)), 2, 2)
        return F.tanh(self.fc1(x.reshape(x.shape[0], -1)))


class SyntheticDataset:
    def __init__(self, samples, shape, seed):
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(-1, 1, (samples, *shape)).astype(np.float32)
        self.t = self.x.mean(axis=(1, 2, 3)).reshape(-1, 1).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.t[i]


class LimitedDataset:
    def __init__(self, dataset, count):
        self.dataset, self.count = dataset, min(count, len(dataset))

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        return self.dataset[i]


def parameters(model):
    values = {}
    model._flatten_params(values)
    return dict(sorted(values.items()))


def snapshot(model):
    return {k: cuda.as_numpy(p.data).copy() for k, p in parameters(model).items()}


def state_hash(state):
    digest = hashlib.sha256()
    for k, v in sorted(state.items()):
        digest.update(k.encode()); digest.update(v.tobytes())
    return digest.hexdigest()


def initialize(model, shape, seed):
    with hytorch.no_grad():
        model(np.zeros((1, *shape), np.float32))
    # Canonical explicit FP32 state; identical weights across frameworks and runs.
    rng = np.random.default_rng(seed)
    for name, p in parameters(model).items():
        if name.endswith("/W"):
            fan_in = p.shape[0] if p.ndim == 2 else np.prod(p.shape[1:])
            p.data = (rng.standard_normal(p.shape).astype(np.float32) * np.float32(1/np.sqrt(fan_in)))
        else:
            p.data = np.zeros(p.shape, np.float32)
    return snapshot(model)


def torch_model(kind, state):
    import torch
    from torch import nn

    class MatchedNet(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "tiny":
                self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
                self.fc1 = nn.Linear(4*8*8, 1)
            else:
                specs = [(3,24,5,2), (24,36,5,2), (36,48,5,2), (48,64,3,1), (64,64,3,1)]
                for i, (ci, co, k, s) in enumerate(specs, 1):
                    setattr(self, f"conv{i}", nn.Conv2d(ci, co, k, stride=s))
                self.fc1, self.fc2, self.fc3 = nn.Linear(1152,100), nn.Linear(100,50), nn.Linear(50,1)

        def forward(self, x):
            if kind == "tiny":
                x = torch.nn.functional.max_pool2d(torch.relu(self.conv1(x)), 2)
                return torch.tanh(self.fc1(x.flatten(1)))
            for i in range(1, 6):
                x = torch.relu(getattr(self, f"conv{i}")(x))
            x = torch.relu(self.fc1(x.flatten(1)))
            return torch.tanh(self.fc3(torch.relu(self.fc2(x))))

    result = MatchedNet()
    with torch.no_grad():
        for key, a in state.items():
            name, param = key.split("/")
            target = getattr(getattr(result, name), "weight" if param == "W" else "bias")
            target.copy_(torch.from_numpy(a.T.copy() if a.ndim == 2 else a))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", choices=["hytorch", "torch"], default="hytorch")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--monitor-id", help="nvidia-smi physical GPU id/UUID/PCI bus; default resolve CUDA PCI bus")
    p.add_argument("--matmul", choices=["cupy","basic","blocked","shared"], default="cupy")
    p.add_argument("--conv", choices=["cupy","direct","im2col"], default="cupy")
    p.add_argument("--pool", choices=["cupy","native"], default="cupy")
    p.add_argument("--tile", type=int, choices=[16,32], default=16)
    p.add_argument("--model", choices=["tiny","donkey"], default="donkey")
    p.add_argument("--data", help="Real DonkeyCar directory; absent means explicitly synthetic smoke test")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=0, help="0 means full epoch; otherwise truncate each epoch")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--mode", choices=["phases","throughput"], default="phases")
    p.add_argument("--interval", type=float, default=1.)
    p.add_argument("--no-monitor", action="store_true")
    p.add_argument("--nvtx", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    native = args.matmul != "cupy" or args.conv != "cupy" or args.pool != "cupy"
    if native and (args.device != "cuda" or args.framework != "hytorch"):
        p.error("Native backend requires --framework hytorch --device cuda")
    if min(args.batch, args.epochs, args.samples) <= 0 or min(args.steps, args.workers, args.warmup) < 0:
        p.error("Invalid batch/epoch/sample/step/worker/warmup value")
    if args.nvtx and (args.device != "cuda" or args.framework != "hytorch"):
        p.error("NVTX ranges here require HyTorch CUDA")
    if args.data and args.model != "donkey":
        p.error("Real DonkeyCar data requires --model donkey")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    shape = (3, 16, 16) if args.model == "tiny" else (3, 66, 200)
    val_ds = None
    if args.data:
        from donkeycar_hy.tub_dataset import DonkeyDataset
        dataset = DonkeyDataset(args.data, augment=False)
        dataset, val_ds = dataset.split(seed=args.seed)
        manifest = [(Path(f).name, float(t), Path(f).stat().st_size)
                    for f, t in zip(dataset._paths, dataset._angles)]
        data_hash = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
        dump_json(out/"train_manifest.json", manifest)
    else:
        dataset = SyntheticDataset(args.samples, shape, args.seed)
        data_hash = hashlib.sha256(dataset.x.tobytes()+dataset.t.tobytes()).hexdigest()
    if len(dataset) == 0:
        p.error("Empty training split")
    if args.model == "tiny":
        hy = TinyNet()
    else:
        from donkeycar_hy.model import DonkeyNet
        hy = DonkeyNet(dropout_rate=0.)
    initial = initialize(hy, shape, args.seed)
    sync = lambda: None
    monitor_id = None
    cp = None
    if args.device == "cuda":
        import cupy as cp
        cp.cuda.Device(args.gpu).use()
        monitor_id = args.monitor_id or cp.cuda.Device(args.gpu).pci_bus_id
        sync = cp.cuda.get_current_stream().synchronize
        if native:
            load_library()
    if args.framework == "hytorch":
        model = hy
        if cp is not None:
            model.to_gpu()
        optimizer = SGD(args.lr).setup(model)
        convert = lambda x: cp.asarray(x) if cp is not None else x
        forward = lambda x,t: F.mean_square_error(model(x), t)
        backward = lambda loss: (model.cleargrads(), loss.backward())
        update = optimizer.update
        restore = lambda: [setattr(v, "data", convert(initial[k].copy())) for k,v in parameters(model).items()]
        scalar = lambda x: float(cuda.as_numpy(x.data))
    else:
        import torch
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = f"cuda:{args.gpu}" if cp is not None else "cpu"
        model = torch_model(args.model, initial).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        convert = lambda x: torch.from_numpy(x).to(device)
        forward = lambda x,t: torch.nn.functional.mse_loss(model(x), t)
        backward = lambda loss: (optimizer.zero_grad(set_to_none=True), loss.backward())
        update = optimizer.step
        saved = {k:v.detach().clone() for k,v in model.state_dict().items()}
        restore = lambda: model.load_state_dict(saved)
        scalar = lambda x: float(x.detach().cpu())
        if cp is not None:
            sync = lambda: torch.cuda.synchronize(args.gpu)
    metadata = dict(vars(args), initial_state_sha256=state_hash(initial), dataset_sha256=data_hash,
                    data_kind="real" if args.data else "SYNTHETIC_SMOKE_NOT_TASK_ACCURACY",
                    sample_order="fixed sequential; shared HyTorch loader for both frameworks",
                    optimizer="SGD, no momentum or weight decay", dropout=0., dtype="float32",
                    python=sys.version, platform=platform.platform(), numpy=np.__version__,
                    data_fingerprint="filename/label/file-size manifest" if args.data else "full array bytes",
                    backward_backend="original", status="RUNNING")
    if cp is not None:
        metadata.update(cupy=cp.__version__, gpu_name=str(cp.cuda.runtime.getDeviceProperties(args.gpu)["name"]),
                        runtime=cp.cuda.runtime.runtimeGetVersion(), physical_monitor_id=monitor_id)
    if args.framework == "torch":
        metadata["torch"] = torch.__version__
    dump_json(out/"config.json", metadata)
    profile = StepProfiler(out/"steps.jsonl", sync, args.mode, args.nvtx)
    monitor = ResourceMonitor(out/"hardware.jsonl", monitor_id, args.interval)
    epochs = []
    context = backend(args.matmul, args.conv, args.pool, args.tile)
    with context:
        # Warmup uses same shapes; restore model, clear grads; SGD has no state.
        warm = [dataset[i] for i in range(min(args.batch, len(dataset)))]
        wx, wt = convert(np.stack([v[0] for v in warm])), convert(np.stack([v[1] for v in warm]))
        for _ in range(args.warmup):
            loss = forward(wx, wt); backward(loss); update()
        sync(); restore(); dispatch_stats(reset=True)
        with monitor if not args.no_monitor else nullcontext():
            for epoch in range(1, args.epochs+1):
                epoch_start = time.perf_counter()
                epoch_dataset = LimitedDataset(dataset, args.steps*args.batch) if args.steps else dataset
                loader = DataLoader(epoch_dataset, args.batch, shuffle=False, num_workers=args.workers)
                iterator = iter(loader)
                limit = (len(dataset)+args.batch-1)//args.batch
                if args.steps:
                    limit = min(limit, args.steps)
                samples, total_loss = 0, 0.
                try:
                    for step in range(1, limit+1):
                        profile.begin()
                        with profile.phase("data"):
                            x,t = next(iterator)
                        with profile.phase("transfer"):
                            x,t = convert(x),convert(t)
                        with profile.phase("forward"):
                            loss = forward(x,t)
                        with profile.phase("backward"):
                            backward(loss)
                        with profile.phase("update"):
                            update()
                        profile.end(len(x),epoch,step)
                        value = scalar(loss)
                        if not np.isfinite(value):
                            raise RuntimeError("Non-finite training loss")
                        total_loss += value*len(x); samples += len(x)
                finally:
                    if hasattr(iterator, "_shutdown_workers"):
                        iterator._shutdown_workers()
                sync()
                train_wall = time.perf_counter()-epoch_start
                record = dict(epoch=epoch, train_loss=total_loss/samples, train_wall_s=train_wall,
                              samples=samples, train_samples_per_sec=samples/train_wall)
                if val_ds is not None:
                    val_start = time.perf_counter()
                    val_total = 0.
                    evaluation = hytorch.no_grad() if args.framework == "hytorch" else torch.no_grad()
                    with evaluation:
                        for x,t in DataLoader(val_ds,args.batch,shuffle=False):
                            val_total += scalar(forward(convert(x),convert(t)))*len(x)
                    record.update(val_loss=val_total/len(val_ds), val_wall_s=time.perf_counter()-val_start)
                epochs.append(record)
                print(json.dumps(record), flush=True)
    hardware = monitor.summary()
    hardware["time_scope"] = "training_and_validation_excluding_warmup"
    timing = profile.summary()
    summary = dict(status="COMPLETED", gpu_status="EXECUTED" if cp is not None else "NOT_EXECUTED",
                   timing=timing, hardware=hardware, epochs=epochs, dispatch_counts=dispatch_stats(),
                   bottleneck_hypotheses=diagnose(timing,hardware))
    if args.framework == "hytorch":
        final = snapshot(model)
    else:
        final = {}
        for key in initial:
            name,param = key.split("/")
            v = getattr(getattr(model,name), "weight" if param == "W" else "bias").detach().cpu().numpy()
            final[key] = v.T.copy() if v.ndim == 2 else v.copy()
    np.savez(out/"final_weights.npz", **final)
    summary["final_state_sha256"] = state_hash(final)
    dump_json(out/"summary.json", summary)


if __name__ == "__main__":
    main()
