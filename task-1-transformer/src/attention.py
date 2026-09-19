import torch
from torch import nn
import math
def scaled_dot_product_attention(Q,K,V,mask=None,drop_layer=None,return_attn=False):
    first_mat=torch.matmul(Q,K.transpose(-2,-1))
    scale_mat=torch.div(first_mat, math.sqrt(K.shape[-1]))
    
    if mask!=None:
        scale_mat=torch.masked_fill(scale_mat,mask,float('-inf'))
    soft_mat=torch.softmax(scale_mat,dim=-1)
    if drop_layer:
        soft_mat=drop_layer(soft_mat)                                    
    out_mat=torch.matmul(soft_mat,V)

    if return_attn:
        return out_mat,soft_mat
    return out_mat
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head,dropout=0.1):
        super().__init__()
        self.n_head=n_head
        assert d_model%n_head==0
        self.d_head=d_model//n_head
        self.W_Q=nn.Linear(d_model,d_model)
        self.W_K=nn.Linear(d_model,d_model)
        self.W_V=nn.Linear(d_model,d_model)
        self.out=nn.Linear(d_model,d_model)
        self.drop=nn.Dropout(dropout)
    def split(self,x):
        B,T,C=x.shape
        self.d_head=C//self.n_head
        x=x.view(B,T,self.n_head,self.d_head)
        return x.transpose(1,2)
    def combine(self,x):
        B,n_head,T,d_head=x.shape
        x=x.transpose(1,2)
        return x.contiguous().view(B,T,n_head*d_head)

    def forward(self,x,mask=None,dropout=None,return_attn=False):
        Q=self.split(self.W_Q(x))
        K=self.split(self.W_K(x))
        V=self.split(self.W_V(x))
        if mask!=None:
            mask=mask.unsqueeze(1).unsqueeze(2)   
        if return_attn:
            notice_out,attn_para=scaled_dot_product_attention(Q,K,V,mask,self.drop,return_attn)
            return attn_para
        else:
            notice_out=scaled_dot_product_attention(Q,K,V,mask,self.drop,return_attn)
        combine_data=self.combine(notice_out)
        out=self.out(combine_data)
        return out

