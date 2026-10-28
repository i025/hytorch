"""Torchvision ImageNet V1 compatible CNNs implemented entirely in HyTorch.

The explicit 224x224 input contract avoids silently substituting a different
adaptive pooling implementation. Torch is needed only by the converter/verifier.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from hytorch import no_grad, cuda
from hytorch.models import Model
import hytorch.layers as L
import hytorch.functions as F

SUPPORTED_MODELS = ("alexnet", "vgg11", "vgg16")
VGG_CONFIGS = {
    "vgg11": [64,"M",128,"M",256,256,"M",512,512,"M",512,512,"M"],
    "vgg16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"],
}


class ImageNetCNN(Model):
    def __init__(self, architecture, num_classes=1000):
        super().__init__()
        if architecture not in SUPPORTED_MODELS or num_classes <= 0:
            raise ValueError("Unsupported architecture or invalid class count")
        self.architecture, self.num_classes = architecture, num_classes
        self.feature_ops, self.weight_map, self.expected_shapes = [], {}, {}
        self.frozen_features = False
        if architecture == "alexnet":
            specs = [(0,3,64,11,4,2), (3,64,192,5,1,2), (6,192,384,3,1,1),
                     (8,384,256,3,1,1), (10,256,256,3,1,1)]
            for idx, ci, co, k, s, pad in specs:
                self._conv(idx,ci,co,k,s,pad)
                if idx in (0,3,10):
                    self.feature_ops.append(("pool",3,2))
            flat, positions = 256*6*6, (1,4,6)
        else:
            ci, idx = 3, 0
            for co in VGG_CONFIGS[architecture]:
                if co == "M":
                    self.feature_ops.append(("pool",2,2)); idx += 1
                else:
                    self._conv(idx,ci,co,3,1,1); ci=co; idx += 2
            flat, positions = 512*7*7, (0,3,6)
        for j,(ni,no,pos) in enumerate(zip((flat,4096,4096),(4096,4096,num_classes),positions),1):
            name = f"fc{j}"
            setattr(self,name,L.Linear(no))
            self.weight_map[f"classifier.{pos}.weight"] = name + "/W"
            self.weight_map[f"classifier.{pos}.bias"] = name + "/b"
            self.expected_shapes[name+"/W"] = (ni,no)
            self.expected_shapes[name+"/b"] = (no,)

    def _conv(self,idx,ci,co,k,s,pad):
        name = f"features_{idx}"
        setattr(self,name,L.Conv2d(co,k,stride=s,pad=pad))
        self.feature_ops.append(("conv",name))
        self.weight_map[f"features.{idx}.weight"] = name+"/W"
        self.weight_map[f"features.{idx}.bias"] = name+"/b"
        self.expected_shapes[name+"/W"] = (co,ci,k,k)
        self.expected_shapes[name+"/b"] = (co,)

    def features(self,x):
        for op in self.feature_ops:
            if op[0] == "conv":
                x = F.relu(getattr(self,op[1])(x))
            else:
                x = F.pooling(x,op[1],op[2])
        return x

    def forward(self,x):
        if x.ndim != 4 or x.shape[1:] != (3,224,224):
            raise ValueError("ImageNetCNN requires preprocessed NCHW (N,3,224,224)")
        if self.frozen_features:
            with no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        if self.architecture == "alexnet":
            x = F.dropout(x)
            x = F.dropout(F.relu(self.fc1(x)))
            x = F.relu(self.fc2(x))
        else:
            x = F.dropout(F.relu(self.fc1(x)))
            x = F.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

    def freeze_features(self, frozen=True):
        self.frozen_features = frozen
        for op in self.feature_ops:
            if op[0] == "conv":
                getattr(self,op[1]).cleargrads()

    def load_pretrained(self,path,replace_head=False,verify_checksum=True):
        path = Path(path)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata.get("architecture") != self.architecture or metadata.get("source") != "torchvision":
            raise ValueError("Weight metadata does not match model / torchvision source")
        if metadata.get("weights") != "IMAGENET1K_V1":
            raise ValueError("Expected ImageNet V1 weight provenance")
        if verify_checksum:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
            if digest.hexdigest() != metadata.get("npz_sha256"):
                raise ValueError("Weight checksum does not match conversion metadata")
        params = {}; self._flatten_params(params)
        with np.load(path, allow_pickle=False) as state:
            if set(state.files) != set(params):
                raise ValueError("Weight keys do not match model")
            # Validate all shapes before mutating the model.
            for key in params:
                expected = self.expected_shapes[key]
                if key.startswith("fc3/") and replace_head:
                    expected = (4096,1000) if key.endswith("/W") else (1000,)
                if state[key].shape != expected or state[key].dtype != np.float32:
                    raise ValueError(f"Invalid weight shape/dtype: {key}")
            for key,param in params.items():
                if replace_head and key.startswith("fc3/"):
                    continue
                param.data = state[key].copy()
        return metadata


def create_model(name, weights=None, num_classes=1000, freeze_features=False):
    model = ImageNetCNN(name,num_classes)
    if weights is not None:
        model.load_pretrained(weights,replace_head=num_classes != 1000)
    model.freeze_features(freeze_features)
    return model


def preprocess(image):
    """ImageNet V1: RGB, bilinear short edge 256, center 224, mean/std."""
    from PIL import Image
    image = image.convert("RGB")
    w,h = image.size
    size = (256,int(256*h/w)) if w <= h else (int(256*w/h),256)
    image = image.resize(size,Image.Resampling.BILINEAR)
    left,top = round((image.width-224)/2),round((image.height-224)/2)
    x = np.asarray(image.crop((left,top,left+224,top+224)),dtype=np.float32)/np.float32(255.)
    x = (x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return np.ascontiguousarray(x.transpose(2,0,1))
