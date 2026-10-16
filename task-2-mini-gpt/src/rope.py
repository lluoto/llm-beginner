import torch
from torch import nn
import math
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class rope(nn.Module):
    def __init__(self, head_dim,base=10000):
        super().__init__()
        freq=1.0/(base**(torch.arange(0,head_dim,2).float()/head_dim))
        self.register_buffer('freq',freq)
    def forward(self,x,offset=0):
        assert torch.isfinite(x).all(), f"rope 输入 NaN, shape={x.shape}, offset={offset}"

        d=x.shape[-1]
        T=x.shape[-2]
        t=torch.arange(offset,offset+T,device=x.device).float()
        freqs=torch.outer(t,self.freq)
        cos=freqs.cos().unsqueeze(0).unsqueeze(0)
        sin=freqs.sin().unsqueeze(0).unsqueeze(0)
        x1,x2=x[...,:d//2],x[...,d//2:]
        return torch.cat([x1*cos - x2*sin,
                          x1 * sin+x2*cos],dim=-1)