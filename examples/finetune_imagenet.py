"""Transfer learning: official converted weights + frozen features + new head.

Input directories: --data/train/<class>/* and --data/val/<class>/*.
"""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hytorch
from hytorch import cuda
import hytorch.functions as F
from hytorch.model_zoo import create_model, preprocess, SUPPORTED_MODELS
from hytorch.dataloaders import DataLoader
from hytorch.optimizers import SGD, FreezeParam
from hytorch.profiling import dump_json


class ImageFolder:
    def __init__(self,path,classes):
        self.records=[]
        for label,name in enumerate(classes):
            self.records.extend((p,label) for p in sorted((Path(path)/name).rglob("*"))
                                if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp"))
        if not self.records:
            raise ValueError(f"No labelled images in {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        from PIL import Image
        path,label=self.records[index]
        with Image.open(path) as im:
            return preprocess(im),np.int64(label)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=SUPPORTED_MODELS,required=True)
    p.add_argument("--weights",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--train-all",action="store_true")
    p.add_argument("--head-only",action="store_true",help="Freeze both features and first two classifier layers")
    args=p.parse_args()
    if args.train_all and args.head_only:
        p.error("Choose --train-all or --head-only")
    if args.epochs <= 0 or args.batch <= 0:
        p.error("epochs and batch must be positive")
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    classes=sorted(p.name for p in (Path(args.data)/"train").iterdir() if p.is_dir())
    if len(classes)<2:
        p.error("At least two class directories required")
    np.random.seed(args.seed)
    train=ImageFolder(Path(args.data)/"train",classes)
    val=ImageFolder(Path(args.data)/"val",classes)
    model=create_model(args.model,args.weights,len(classes),freeze_features=not args.train_all)
    # Always explicitly initialize the replacement head in FP32, including Numpy 2.
    model.fc3.in_size=4096
    model.fc3.W.data=(np.random.randn(4096,len(classes))*np.sqrt(1/4096)).astype(np.float32)
    optimizer=SGD(args.lr).setup(model)
    if args.head_only:
        optimizer.add_hook(FreezeParam(model.fc1,model.fc2))
    if args.device=="cuda":
        model.to_gpu()
    convert=lambda x:cuda.as_cupy(x) if args.device=="cuda" else x
    before={}; model._flatten_params(before)
    # Only store frozen parameters' hashes, not a second model-size copy.
    import hashlib
    frozen={k:hashlib.sha256(cuda.as_numpy(v.data).tobytes()).hexdigest()
            for k,v in before.items() if (not args.train_all and k.startswith("features_"))
            or (args.head_only and k.startswith(("fc1/","fc2/")))}
    logs=[]
    for epoch in range(1,args.epochs+1):
        total=0.
        with hytorch.using_config("train",True):
            for x,t in DataLoader(train,args.batch,shuffle=True):
                loss=F.softmax_cross_entropy(model(convert(x)),convert(t))
                model.cleargrads(); loss.backward(); optimizer.update()
                total+=float(cuda.as_numpy(loss.data))*len(x)
        correct=0
        with hytorch.no_grad(),hytorch.test_mode():
            for x,t in DataLoader(val,args.batch,shuffle=False):
                prediction=cuda.as_numpy(model(convert(x)).data).argmax(axis=1)
                correct+=int((prediction==t).sum())
        logs.append(dict(epoch=epoch,train_loss=total/len(train),val_accuracy=correct/len(val)))
        print(logs[-1],flush=True)
    for k,expected in frozen.items():
        if hashlib.sha256(cuda.as_numpy(before[k].data).tobytes()).hexdigest()!=expected:
            raise AssertionError(f"Frozen parameter changed: {k}")
    state={}; model._flatten_params(state)
    np.savez(out/"finetuned.npz",**{k:cuda.as_numpy(v.data) for k,v in state.items()})
    dump_json(out/"training.json",dict(config=vars(args),classes=classes,epochs=logs,
                                       frozen_parameters_verified=len(frozen)))


if __name__=="__main__":
    main()
