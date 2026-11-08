# hytorch/transformer/feedforward.py
from hytorch import Layer
import hytorch.layers as L
import hytorch.functions as F


class PositionwiseFeedForward(Layer):
    """
    FFN: (..., D) -> (..., D)
    结构: LinearND(D->Dff) -> ReLU -> LinearND(Dff->D)
    """
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = L.LinearND(d_model, d_ff)
        self.fc2 = L.LinearND(d_ff, d_model)
        self.dropout = L.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
