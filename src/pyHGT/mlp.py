import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_h, out_h):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_h, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, out_h)
        #self.linear4 = nn.Linear(32, 2)
        self.silu = nn.SiLU()
        #self.sigm = nn.Sigmoid()
        #self.leak = nn.LeakyReLU()
        #self.soft = nn.Softmax()
        #self.dropout = nn.Dropout(0.2)
        #self.batchnorm1 = nn.BatchNorm1d(64)
        #self.batchnorm2 = nn.BatchNorm1d(32)
        #self.layernorm1 = nn.LayerNorm(64)
        #self.layernorm2 = nn.LayerNorm(32)

    def forward(self, x):
        #out_1 = self.silu(self.layernorm1(self.linear1(x)))
        #out_2 = self.silu(self.layernorm2(self.linear2(out_1)))
        out_1 = self.silu(self.linear1(x))
        out_2 = self.silu(self.linear2(out_1))
        out_3 = self.silu(self.linear3(out_2))
        #out_2 = self.dropout(out_2)  # Add dropout
        out_4 = self.linear4(out_3)
        #out_3 = self.sigm(self.linear3(out_2))
        #out_3 = self.soft(self.linear3(out_2))
        #out_4 = self.sigm(self.linear3(out_3))
        #out = self.silu(self.batchnorm1(self.linear1(x)))
        #out = self.silu(self.batchnorm2(self.linear2(out)))
        #out = self.dropout(out)
        #out = self.soft(self.linear3(out))
        #out_1 = self.leak(self.linear1(x))
        #out_2 = self.leak(self.linear2(out_1))
        #out_3 = self.linear3(out_2)

        return out_4, out_3, out_2, out_1
