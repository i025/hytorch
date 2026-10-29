"""Minimal original Conv2DGradW second-derivative reproduction, CPU only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
from hytorch import Variable
import hytorch.functions as F
from hytorch.functions_conv import conv2d

x = Variable(np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3)/20)
w = Variable(np.ones((2, 2, 2, 2), np.float32)/10)
y = conv2d(x, w)
F.sum(y*y).backward(create_graph=True)
gw = w.grad
x.cleargrad()
w.cleargrad()
F.sum(gw*gw).backward()
print('second derivative finite:', bool(np.isfinite(w.grad.data).all()))
