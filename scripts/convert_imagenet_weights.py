"""Download official torchvision ImageNet weights and convert to HyTorch NPZ."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hytorch.model_zoo import ImageNetCNN, SUPPORTED_MODELS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=SUPPORTED_MODELS,required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--verify",action="store_true",help="Compare eval logits on fixed input after conversion")
    p.add_argument("--verify-device",choices=["cpu","cuda"],default="cpu")
    args=p.parse_args()
    import torch
    import torchvision
    from torchvision import models
    from hytorch import no_grad, test_mode, cuda
    target=Path(args.output)
    if target.suffix != ".npz" or target.exists() or target.with_suffix(".json").exists():
        p.error("Use a new .npz output path")
    weights=models.get_model_weights(args.model).IMAGENET1K_V1
    source=models.get_model(args.model,weights=weights).eval()
    model=ImageNetCNN(args.model)
    state={}
    for key,dest in model.weight_map.items():
        value=source.state_dict()[key].detach().cpu().numpy()
        state[dest]=np.ascontiguousarray(value.T if value.ndim==2 else value,dtype=np.float32)
    if set(state) != set(model.expected_shapes):
        raise RuntimeError("Incomplete model conversion")
    for key,value in state.items():
        if value.shape != model.expected_shapes[key]:
            raise RuntimeError(f"Shape mismatch for {key}")
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez(target,**state)
    digest=hashlib.sha256()
    with target.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            digest.update(block)
    metadata=dict(architecture=args.model,source="torchvision",weights="IMAGENET1K_V1",
                  source_url=weights.url,torch=torch.__version__,torchvision=torchvision.__version__,
                  npz_sha256=digest.hexdigest(),input_shape=[1,3,224,224],
                  verification="NOT_EXECUTED",mapping=model.weight_map)
    meta_path=target.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    if args.verify:
        model.load_pretrained(target)
        x=np.random.default_rng(42).standard_normal((1,3,224,224)).astype(np.float32)
        device=args.verify_device
        source=source.to(device)
        if device=="cuda":
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            model.to_gpu()
        with torch.no_grad():
            expected=source(torch.from_numpy(x).to(device)).cpu().numpy()
        with no_grad(),test_mode():
            actual=cuda.as_numpy(model(cuda.as_cupy(x) if device=="cuda" else x).data)
        metadata.update(max_abs_error=float(np.max(np.abs(actual-expected))),
                        verification="PASSED" if np.allclose(actual,expected,rtol=1e-4,atol=1e-4) else "FAILED",
                        verification_device=device)
        meta_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":
    main()
