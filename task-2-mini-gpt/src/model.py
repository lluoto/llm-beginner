from src.tokenizer import BPETokenizer
from torch import nn
import torch
import copy
from src.attention import MultiHeadAttention
import json
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def nn_duplicate(layer,N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

class decoderlayer(nn.Module):
    def __init__(self,d_model,n_head,d_ff,dropout=0.1):
        super().__init__()
        self.mask_att=MultiHeadAttention(d_model,n_head,dropout)
        self.fnn=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        self.drop=nn.Dropout(dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
    def forward(self,x,mask=None,kv_cache=None,return_cache=False):
        out,kv_cache=self.mask_att(self.norm1(x),mask,kv_cache,return_cache)
        x=x+self.drop(out)
        x=x+self.norm2(self.fnn(self.drop(x)))
        if return_cache:
            return x,kv_cache
        return x

class decoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        self.layer=nn_duplicate(layer,N)
        self.layer_num=N
    def forward(self,x,mask=None,kv_cache=None,return_cache=False):
        for index,layer in enumerate(self.layer):
            if return_cache:
                if kv_cache is None:
                    kv_cache={i:[None,None] for i in range(self.layer_num)}
                x,kv_cache[index]=layer(x,mask,kv_cache[index],return_cache)
            else:
                x=layer(x,mask,kv_cache,return_cache)
        if return_cache:
            return x,kv_cache
            
        return x

class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_head,N=6,d_ff=2048,vocab_len=2500,dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.n_head=n_head
        self.decoder=decoder(decoderlayer(d_model,n_head,d_ff,dropout),N)
        self.output_embedding=nn.Embedding(vocab_len,embedding_dim=d_model)
        self.decode_linear=nn.Linear(d_model,vocab_len)
    def forward(self,output_token,mask=None,kv_cache=None,return_cache=False):
        output_embed=self.output_embedding(output_token)
        if return_cache:
            decoder_out,kv_cache=self.decoder(output_embed,mask,kv_cache,return_cache)
        else:
            decoder_out=self.decoder(output_embed,mask,kv_cache,return_cache)
        
        lin_out=self.decode_linear(decoder_out)
        if return_cache:
            return lin_out,kv_cache
        return lin_out

class MiniGPT(nn.Module):
    def __init__(self,d_model,n_head,d_ff,word2idx,pad_token_id=0,unk_token_id=1,N=6,vocab_len=512,dropout=0.1):
        super().__init__()
        self.word2idx=word2idx
        self.pad_token_id=pad_token_id
        self.unk_token_id=unk_token_id
        assert d_model%n_head==0
        self.block=TransformerBlock(d_model,n_head,N,d_ff,vocab_len,dropout)
        self.max_len=vocab_len
        self.kv_cache=None
    def forward(self,ids,mask=None,kv_cache=None,return_cache=False):
        ids=ids.to(self.block.output_embedding.weight.device)
        
        
        if return_cache:
            x,kv_cache=self.block(ids,mask,kv_cache,return_cache)
            return x,kv_cache

        else:
            if mask is None:
                mask=(ids==self.pad_token_id)
            x=self.block(ids,mask,kv_cache,return_cache)
            return x
    def generate(self,prompt_ids,temperature,top_k=None,top_p=None,mask=None,
                 return_cache=False,max_len=64):

        self.reset_kv_cache()
        max_len=max_len or self.max_len
        # forward 契约：return_cache=True 时返回 (logits, cache)，否则只返回 logits
        out=self.forward(prompt_ids,mask,self.kv_cache,return_cache)
        if return_cache:
            logits,self.kv_cache=out
        else:
            logits=out
        generated=torch.empty(prompt_ids.size(0),0,dtype=prompt_ids.dtype,device=prompt_ids.device)
        while generated.size(1)<max_len:
            next_logits=logits[:,-1,:].clone()
            next_logits[:,self.pad_token_id]=float('-inf')

            if temperature==0:
                next_id=next_logits.argmax(dim=-1,keepdim=True)
            else:
                next_logits=next_logits/temperature

                if top_k:
                    next_logits,topk_indice=torch.topk(next_logits,top_k,dim=-1)
                if top_p:
                    sort_logits,topp_idx=torch.sort(next_logits,dim=-1,descending=True)
                    prob=torch.softmax(sort_logits,dim=-1)
                    accumulate_prob=torch.cumsum(prob,dim=-1)
                    keep=accumulate_prob<=top_p
                    keep[...,1:]=keep[...,1:]|keep[...,:-1]
                    keep[...,0]=1
                    next_logits=torch.where(keep,sort_logits,float('-inf'))
                prob=torch.softmax(next_logits,dim=-1)
                next_id=torch.multinomial(prob,1)
                # 采样下标先经 top_p 还原到 top_k 下标，再经 top_k 还原到词表下标；
                # 顺序必须是 top_p -> top_k，否则两者同时使用时下标会越界。
                if top_p:
                    next_id=topp_idx.gather(-1,next_id)
                if top_k:
                    next_id=topk_indice.gather(-1,next_id)

            generated=torch.cat([generated,next_id],dim=1)
            if generated.size(1)>=max_len:
                break
            if return_cache:
                logits,self.kv_cache=self.forward(next_id,mask,self.kv_cache,True)
            else:
                # 不开 cache：每步喂 prompt+已生成，等价于 O(T^2) 全量前向
                logits=self.forward(torch.cat([prompt_ids,generated],dim=1),mask,None,False)
        return torch.cat([prompt_ids,generated],dim=1)
    def reset_kv_cache(self):
        self.kv_cache=None
    
def load_for_eval(ckpt_path):
    with open('ckpt/dimention_parameter.json','r')as f:
        config=json.load(f)
    BPE_processor=BPETokenizer.from_pretrained('ckpt/tokenizer.json',128)

    model = MiniGPT(
        d_model=config['dimention_of_model'],          # 或 args.dimention_of_model
        n_head=config['number_head'],             # 或 args.number_head
        N=config['layer_number'],                  # 或 args.block_number
        d_ff=config['dimention_of_FFN'],             # 或 args.dimention_of_FFN
        word2idx=BPE_processor.word2idx,        
        vocab_len=len(BPE_processor.word2idx)      # 或 args.vocab_len
    )

    # 2. 加载权重
    model.load_state_dict(torch.load(ckpt_path))
    model=model.to(device)
    # 3. 评估模式
    model.eval()
    return model,BPE_processor