import matplotlib.pyplot as plt
from matplotlib import font_manager
from torch.nn.modules.loss import CrossEntropyLoss
import argparse
import torch
import random
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from src.model import TransformerClassifier
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from collections import Counter
import time
import gc
import math
import os 
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def pad_mask_generate(input_tensor,pad_token_id=0):
    pad_mask = (input_tensor == pad_token_id)
    return pad_mask    


def set_seed(seed=1111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_personal_vocab(tokens,vocab_len=2500):
    counter=Counter()
    for text in tokens:
        counter.update(list(text))
    special_token=['<pad>','<unk>','<bos>','<eos>']
    chars=[c for c,_ in counter.most_common(vocab_len-len(special_token))]
    vocab=special_token+chars
    word2idx={w:i for i,w in enumerate(vocab)}
    idx2word={i:w for i,w in enumerate(vocab)}
    return vocab,word2idx,idx2word

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_attention(model,text,layer_idx=0,head_idx=0,max_len=512):
    """返回第 layer_idx 层、第 head_idx 个头的注意力矩阵(已裁剪到真实序列长度)及对应字符。"""
    model.eval()
    chars=list(text)[:max_len]
    real_len=max(1,len(chars))
    with torch.no_grad():
        input_token=model.tokenizer(text,max_len).to(device).unsqueeze(0)
        mask=pad_mask_generate(input_token,0).to(device)
        x=model.block.input_post_embed(model.block.input_embedding(input_token))
        for count,layer in enumerate(model.block.encoder.layer):
            if count==layer_idx:
                attn=layer.multihead(x,mask=mask,dropout=None,return_attn=True)
                break
            x=layer(x,mask)   # 前面各层同样施加 padding mask,避免 pad 信息泄漏
    # attn: (B, n_head, T, T) -> 取该 head 并裁剪到真实 token 数
    attn_mat=attn[0,head_idx,:real_len,:real_len].cpu().numpy()
    return attn_mat,chars

def setup_cjk_font():
    """尽量启用中文字体;若系统无 CJK 字体则返回 None,调用方退化为用位置索引做刻度。"""
    candidates=['Noto Sans CJK SC','Noto Serif CJK SC','Source Han Sans SC',
                'WenQuanYi Zen Hei','WenQuanYi Micro Hei','Microsoft YaHei',
                'SimHei','AR PL UMing CN','Droid Sans Fallback']
    available={f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif']=[name]
            plt.rcParams['axes.unicode_minus']=False
            return name
    return None

def pick_viz_samples(df,max_label_len=40):
    """从数据集中挑选 正面/负面/长句 各一条用于注意力可视化。"""
    d=df.copy()
    d['_len']=d['text'].str.len()
    samples=[]
    pos=d[(d['label']==1)&(d['_len'].between(6,max_label_len))].sort_values('_len')
    neg=d[(d['label']==0)&(d['_len'].between(6,max_label_len))].sort_values('_len')
    lon=d[d['_len'].between(60,200)].sort_values('_len')
    if len(pos): samples.append(('pos',pos.iloc[0]['text']))
    if len(neg): samples.append(('neg',neg.iloc[0]['text']))
    if len(lon): samples.append(('long',lon.iloc[len(lon)//2]['text']))
    return samples

def plot_attention_grid(model,text,tag,layer_number,n_head,out_dir='figures',
                        cjk_ok=True,max_label_len=40):
    """画一张 (选定层 × 所有头) 的注意力网格图,坐标轴用字符标注,附 colorbar。"""
    layers=sorted(set([0,layer_number//2,layer_number-1]))
    os.makedirs(out_dir,exist_ok=True)
    fig,axes=plt.subplots(len(layers),n_head,
                          figsize=(2.6*n_head,2.8*len(layers)),squeeze=False)
    im=None
    for r,j in enumerate(layers):
        for c in range(n_head):
            attn_mat,chars=get_attention(model,text,layer_idx=j,head_idx=c)
            n=len(chars)
            ax=axes[r][c]
            im=ax.imshow(attn_mat,vmin=0,vmax=1,cmap='viridis',aspect='auto')
            ax.set_title(f'L{j} H{c}',fontsize=9)
            show_tick = n<=max_label_len          # 太长就不标刻度,否则挤成一团
            labels = chars if cjk_ok else [str(t) for t in range(n)]
            if show_tick and c==0:
                ax.set_yticks(range(n)); ax.set_yticklabels(labels,fontsize=6)
            else:
                ax.set_yticks([])
            if show_tick and r==len(layers)-1:
                ax.set_xticks(range(n)); ax.set_xticklabels(labels,fontsize=6,rotation=90)
            else:
                ax.set_xticks([])
    preview=text[:20]+('...' if len(text)>20 else '')
    fig.suptitle(f'Attention · {tag} · "{preview}"',fontsize=11)
    fig.colorbar(im,ax=axes,shrink=0.6,label='attention weight')
    path=f'{out_dir}/attn_grid_{tag}.png'
    fig.savefig(path,dpi=150,bbox_inches='tight')
    plt.close(fig)
    return path
class read_dataset():
    def __init__(self,file_dir,word2idx,model,max_len=512,pad_token_id=0):
        super().__init__()
        df=pd.read_parquet(file_dir)
        self.label=df['label'].tolist()#[:32]
        self.text=df['text'].tolist()#[:32]
        self.model=model
        self.max_len=max_len
        self.pad_token_id=pad_token_id
    def __len__(self):
        return len(self.label)
    def __getitem__(self,index):

        text=self.text[index]
        input_tensor=self.model.tokenizer(text,self.max_len)

        label_tensor=torch.tensor(self.label[index],dtype=torch.long)
        return input_tensor,label_tensor
        

def parser_para():
    parser=argparse.ArgumentParser(description='Transformer for emotion labeling')
    parser.add_argument('--train_dir','-train',type=str,default='data/train.parquet')
    parser.add_argument('--valid_dir','-valid',type=str,default='data/validation.parquet')
    parser.add_argument('--test_dir','-test',type=str,default='data/test.parquet',required=False)
    parser.add_argument('--epochs','-e',type=int,default='10')
    parser.add_argument('--batch_size','-b',type=int,default='16')
    parser.add_argument('--dimention_of_model','-dmodel',type=int,default='128')#128
    parser.add_argument('--dimention_of_FFN','-dff',type=int,default='512')
    parser.add_argument('--dropout_rate','-d',type=float,default='0.15')
    parser.add_argument('--layer_number','-l',type=int,default='9')#6
    parser.add_argument('--seed','-s',type=int,default='114514')

    parser.add_argument('--learning_rate','-lr',type=float,default='3e-4')
    parser.add_argument('--max_len','-max',type=int,default='512')
    parser.add_argument('--vocab_len','-v',type=int,default='2500')
    parser.add_argument('--number_head','-nh',type=int,default='4')
    return parser
#accuracy:0.8716666666666667


def main():
    accumulate_step=0
    args=parser_para().parse_args()

    epochs=args.epochs
    df=pd.read_parquet('data/train.parquet')
    text=df['text'].tolist()
    vocab,word2idx,idx2word=build_personal_vocab(text)
    
    for _ in range(10):
        writer=SummaryWriter()

        current_seed=args.seed
        
        if args.seed==114514:
            current_seed=random.randint(0,1000002)
        
        set_seed(current_seed)
        start_time=time.time()
        
        model=TransformerClassifier(args.dimention_of_model,args.number_head,
                                    args.dimention_of_FFN,word2idx,
                                    N=args.layer_number,
                                    vocab_len=args.vocab_len,max_len=args.max_len)
        train_set=DataLoader(dataset=
        read_dataset(args.train_dir,word2idx,model,args.max_len)
                        ,batch_size=args.batch_size,shuffle=True)
        train_step=len(train_set)
        ten_precent=train_step//5
        total_step=epochs*train_step
        model=model.cuda(device)
        model.train()

        optimizer=AdamW(model.parameters(),args.learning_rate,weight_decay=0.01)
        warmup=0.2*total_step
        def lambda_para(step):
            if step<warmup:
                return step/warmup
            else:
                progress=(step-warmup)/max(1,total_step-warmup)
                return 0.5*(1+math.cos(math.pi*progress))
        lr_scheduler=LambdaLR(optimizer,lambda_para)
        loss=CrossEntropyLoss()
        loss_fn=loss.to(device)

        for i in tqdm(range(epochs)):
            print(f'===== epoch {i} onset =====')
            step=0
            for batch in tqdm(train_set,desc='everystep'):
                input,target=batch
                input=input.to(device)
                pad_mask=pad_mask_generate(input,0)
                pad_mask=pad_mask.to(device)

             
                target=target.to(device)
                output=model(input,pad_mask)
                curr_loss=loss_fn(output,target)
                optimizer.zero_grad()
                curr_loss.backward()
                clip_grad_norm_(model.parameters(),max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()

                step+=1
                accumulate_step+=1
                if step % ten_precent==0:
                    print(f"\nStep {step} finished, loss: {curr_loss.item()}\n")
                    writer.add_scalar('train_loss',curr_loss.item(),accumulate_step)
                    for name,param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'grad/{name}',param.grad,accumulate_step)
            torch.save(model.state_dict(),f'ckpt/tsf_{i}.pt')
            print(f'saved ckpt/tsf_{i}.pth')
        
            
        hparams={'dimention_of_model':args.dimention_of_model,
                'dimention_of_FFN':args.dimention_of_FFN,
                'number_head':args.number_head,
                'layer_number':args.layer_number,
                'learning_rate':args.learning_rate,
                'dropout_rate':args.dropout_rate,
                'seed':str(current_seed),#+"drop_first_residue",
                'time comsumption':time.time()-start_time
                }
        total_loss=0
        model.eval()

        valid_set=DataLoader(dataset=read_dataset(args.test_dir,word2idx,model),batch_size=args.batch_size)
        with torch.no_grad():
            correct,total=0,0
            for batch in valid_set:
                input,target=batch
                input=input.to(device)
                pad_mask=pad_mask_generate(input,0)
                pad_mask=pad_mask.to(device)
                target=target.to(device)
                

                output=model(input,pad_mask)
                curr_loss=loss_fn(output,target)
                total_loss+=curr_loss.item()
                pred = output.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        accuracy=correct/total
        metrics={'total_loss':total_loss,
                'accuracy':accuracy}
        print(f'total loss:{total_loss},accuracy:{accuracy}')
        writer.add_hparams(hparams,metrics,run_name='.')

        cjk=setup_cjk_font()
        if cjk is None:
            print('[warn] 未找到中文字体,注意力热图坐标轴将用位置索引代替字符(建议安装 fonts-noto-cjk)。')
        for tag,sample_text in pick_viz_samples(pd.read_parquet(args.test_dir)):
            saved=plot_attention_grid(model,sample_text,tag,args.layer_number,
                                      args.number_head,out_dir='figures',cjk_ok=(cjk is not None))
            print(f'saved {saved}  ({tag}: {sample_text[:20]})')
        del model                    # 删除模型
        torch.cuda.empty_cache()     # 清空缓存
        gc.collect() 
        if args.seed!=114514:
            break
        

if __name__=='__main__':
    main()