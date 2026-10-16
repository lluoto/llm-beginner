import torch
from torch import nn
import math
from src.rope import rope
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def scaled_dot_product_attention(Q,K,V,mask=None,drop_layer=None):
    first_mat=torch.matmul(Q,K.transpose(-2,-1))

    scale_mat=torch.div(first_mat, math.sqrt(K.shape[-1]))
    if mask is not None:
        scale_mat=torch.masked_fill(scale_mat,mask,float('-inf'))
    soft_mat=torch.softmax(scale_mat,dim=-1)
    if drop_layer:
        soft_mat=drop_layer(soft_mat)                                    
    out_mat=torch.matmul(soft_mat,V)
    return out_mat
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head,dropout=0.1):
        super().__init__()
        self.n_head=n_head
        assert d_model%n_head==0
        self.d_head=d_model//n_head
        self.Q_rope=rope(self.d_head)
        self.K_rope=rope(self.d_head)

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
    def forward(self,x,mask=None,KV_cache=None,return_cache=False):
        Q=self.split(self.W_Q(x))
        K=self.split(self.W_K(x))
        V=self.split(self.W_V(x))
        assert torch.isfinite(Q).all(), "Q 含 NaN"

        if return_cache:
            
            K_cache,V_cache=KV_cache
            if K_cache is not None:
                cur_pos=K_cache.shape[2]
                Q=self.Q_rope(Q,offset=cur_pos)
                K=self.K_rope(K,offset=cur_pos)
                K=torch.cat([K_cache,K],dim=2)
                V=torch.cat([V_cache,V],dim=2)

            else:
                Q=self.Q_rope(Q,offset=0)
                K=self.K_rope(K,offset=0)
            KV_cache=[K,V]
            
            Q_last=Q[:,:,-1:,:]
            notice_out=scaled_dot_product_attention(Q_last,K,V,None,self.drop)
        else:
            Q=self.Q_rope(Q)
            K=self.K_rope(K)
            if mask is not None:
                L=mask.shape[-1]
                casual_mask=torch.triu(torch.ones(L,L,device=x.device),diagonal=1).bool()
                mask=mask.unsqueeze(1).unsqueeze(2)   
                casual_mask=casual_mask.unsqueeze(0).unsqueeze(0)
                mask=casual_mask|mask
                
            notice_out=scaled_dot_product_attention(Q,K,V,mask,self.drop)
        combine_data=self.combine(notice_out)
        out=self.out(combine_data)
        if KV_cache is not None:
            return out,KV_cache
        return out,None

