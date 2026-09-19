import math
from torch import nn
import copy
from src.attention import MultiHeadAttention
import torch

class encoderlayer(nn.Module):
    def __init__(self,d_model,n_head,d_ff,dropout=0.1):
        super().__init__()
        self.multihead=MultiHeadAttention(d_model,n_head,dropout)
        
        self.fnn=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.drop=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        att_out=self.multihead(x,mask)
        firstnorm=self.norm1(x+self.drop(att_out))
        secondnorm=self.norm2(firstnorm+self.drop(self.fnn(firstnorm)))
        return secondnorm
def nn_duplicate(layer,N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
class encoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        self.layer=nn_duplicate(layer,N)
    def forward(self,x,mask=None):
        for layer in self.layer:
            x=layer(x,mask)
        return x

class positionalencoding(nn.Module):
    def __init__(self,max_len, d_model):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1).float()
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x):
        return x+self.pe[:,:x.size(1)]
class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_head,N=6,d_ff=2048,vocab_len=2500,max_len=512):
        super().__init__()
        self.d_model=d_model
        self.n_head=n_head
        self.encoder=encoder(encoderlayer(d_model,n_head,d_ff),N)
        self.input_embedding=nn.Embedding(vocab_len,embedding_dim=d_model)
        self.input_post_embed=positionalencoding(max_len,self.d_model)
        
    def forward(self,input_token,input_mask=None):
        input_embed=self.input_embedding(input_token)
        input_post_code=self.input_post_embed(input_embed)
        encoder_out=self.encoder(input_post_code,input_mask)
        

        return encoder_out