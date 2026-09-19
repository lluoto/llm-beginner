from torch import nn
from src.block import TransformerBlock
import torch
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_attention(model,text,chartokenizer,word2idx,layer_idx=0,head_idx=0):
    model.eval()
    with torch.no_grad():
        tokenizer=chartokenizer(word2idx)
        input_id=tokenizer.encode(text,max_len=512)
        input_token=torch.tensor(input_id,dtype=torch.long).to(device).unsqueeze(0)
        x=model.block.input_post_embed(model.block.input_embedding(input_token))
        for count,layer in enumerate(model.block.encoder.layer):
            if count==layer_idx:
                attn=layer.multihead(x,mask=None,dropout=None,return_attn=True)
                break
            x=layer(x)
    return attn[0,head_idx].cpu().numpy()
class TransformerClassifier(nn.Module):
    def __init__(self,d_model,n_head,d_ff,word2idx,pad_token_id=0,unk_token_id=1,N=6,vocab_len=2500,max_len=512):
        super().__init__()
        self.word2idx=word2idx
        self.pad_token_id=pad_token_id
        self.unk_token_id=unk_token_id
        assert d_model%n_head==0
        self.block=TransformerBlock(d_model,n_head,N,d_ff,vocab_len,max_len)
        self.classifier=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,2),
        )
    def forward(self,x,mask=None):
        if mask is None:
            mask = (x == self.pad_token_id)
        x=self.block(x,input_mask=mask)
        mask = mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(x)
    def tokenizer(self,text,max_len=512):
        ids=[self.word2idx.get(c,self.unk_token_id)for c in list(text)]
        len_ids=len(ids)
        if len_ids>max_len:
            ids=ids[:max_len]
        else:
            ids+=[self.pad_token_id]*(max_len-len_ids)
            
        input_tensor=torch.tensor(ids,dtype=torch.long)
        return input_tensor
def load_for_eval(ckpt_path,word2idx):
    model = TransformerClassifier(
        d_model=128,          # 或 args.dimention_of_model
        n_head=4,             # 或 args.number_head
        N=9,                  # 或 args.block_number
        d_ff=512,             # 或 args.dimention_of_FFN
        word2idx=word2idx,
        vocab_len=2500,       # 或 args.vocab_len
        max_len=512           # 或 args.max_len
    )

    # 2. 加载权重
    model.load_state_dict(torch.load(ckpt_path))

    # 3. 评估模式
    model.eval()
    return model,lambda text:model.tokenizer(text)