import shutil
from torch.nn.modules.loss import CrossEntropyLoss
import argparse
import torch
import random
from src.tokenizer import BPETokenizer
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.model import MiniGPT
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import time
import json
import gc
import math
import os 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def pad_mask_generate(input_tensor,pad_token_id=0):
    pad_mask = (input_tensor == pad_token_id)
    return pad_mask    # 非 padding 位置为 True
def set_seed(seed=1111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

dataset='simple'
# dataset='skypile'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class read_dataset():
    def __init__(self,file_dir,model,max_len=512,pad_token_id=0):
        super().__init__()
        text=open(file_dir,'r',encoding='utf-8').read()
        self.text=text#.split('\n\n')
        self.model=model
        self.max_len=max_len
        self.pad_token=pad_token_id
        self.block=[]
        # for paragraph in self.text:
        input_tensor=torch.tensor(self.model.encode(self.text,self.max_len),dtype=torch.long)

            # if len(input_tensor)<=self.max_len:
            #     padded=torch.cat([
            #         input_tensor,
            #         torch.full((self.max_len-len(input_tensor),),self.pad_token,dtype=torch.long)
            #     ])
            #     self.block.append(padded)
            # else:
                # comnbine=
        stride=self.max_len
        if dataset=='simple':
            stride=self.max_len//4

        for i in range(0,len(input_tensor)-self.max_len,stride):
            windows=input_tensor[i:i+self.max_len]
            if len(windows)<2:
                break
            self.block.append(windows)
    def __len__(self):
        return len(self.block)
    def __getitem__(self,index):
        
        return self.block[index]
def parser_para():
    parser=argparse.ArgumentParser(description='Transformer for emotion labeling')
    if dataset=='skypile':
        parser.add_argument('--train_dir','-train',type=str,default='data/train.txt')
        parser.add_argument('--test_dir','-test',type=str,default='data/test.txt')
    else:
        parser.add_argument('--train_dir','-train',type=str,default='data/simple/train.txt')
        parser.add_argument('--test_dir','-test',type=str,default='data/simple/poetry.txt')
    parser.add_argument('--epochs','-e',type=int,default='50')
    parser.add_argument('--batch_size','-b',type=int,default='16')
    parser.add_argument('--dimention_of_model','-dmodel',type=int,default='256')# pass:8
    parser.add_argument('--dimention_of_FFN','-dff',type=int,default='1024')#pass : 32
    parser.add_argument('--dropout_rate','-d',type=float,default='0.1')
    parser.add_argument('--layer_number','-l',type=int,default='12')#62 *256*1024 =50 M 12 * 256*1024=10M
    parser.add_argument('--seed','-s',type=int,default='114514')
    parser.add_argument('--cosine_temp','-cs',type=float,default='0.1')
    parser.add_argument('--learning_rate','-lr',type=float,default='4e-3')
    parser.add_argument('--max_len','-max',type=int,default='256')
    parser.add_argument('--vocab_size','-vocab',type=int,default='512')
    parser.add_argument('--number_head','-nh',type=int,default='1')
    parser.add_argument('--relative_rate','-rr',type=float,default='0.05')
    parser.add_argument('--patience','-p',type=int,default='15')
    parser.add_argument('--weight_decay','-wd',type=float,default=0.08)
    return parser


def main():
    args=parser_para().parse_args()
    epochs=args.epochs
    BPE_processor=BPETokenizer(128,args.vocab_size,args.train_dir)
    word2idx,vocab=BPE_processor.word2idx,BPE_processor.vocab
    os.makedirs('ckpt/', exist_ok=True)
    BPE_processor.save_pretrained('ckpt')
    start_time=time.time()
    
    # for _ in range(10):
    # for layer_para in [11,62,125]:
        
    accumulate_step=0
    best_ppl=float('inf')
    current_seed=args.seed
    if args.seed==114514:
        current_seed=random.randint(0,1000002)
    set_seed(current_seed)
    runtime_writer=SummaryWriter(f'runs/{dataset}/seed_{current_seed}')

    hparams={'dimention_of_model':args.dimention_of_model,
            'dimention_of_FFN':args.dimention_of_FFN,
            'number_head':args.number_head,
            'layer_number':args.layer_number,#layer_para,
            'learning_rate':args.learning_rate,
            'dropout_rate':args.dropout_rate,
            'seed':str(current_seed)+' '+dataset,
            'cosine temp':args.cosine_temp,
            'relative_rate':args.relative_rate,
            'patience':args.patience,
            'weight_decay':args.weight_decay
            }
    train_set=DataLoader(dataset=
    read_dataset(args.train_dir,BPE_processor,args.max_len)
                    ,batch_size=args.batch_size,shuffle=True)
    train_step=len(train_set)
    ten_precent=max(float(train_step)//10,1)
    total_step=epochs*train_step
    model=MiniGPT(args.dimention_of_model,args.number_head,
            args.dimention_of_FFN,word2idx,
            N=args.layer_number,#layer_para,
            vocab_len=len(word2idx),dropout=args.dropout_rate)
    model=model.cuda(device)
    model.train()
    optimizer=AdamW(model.parameters(),args.learning_rate,weight_decay=args.weight_decay)
    warmup=args.cosine_temp*total_step
    def lambda_para(step):
        if step<warmup:
            return step/warmup
        else:
            progress=(step-warmup)/max(1,total_step-warmup)
            return 0.5*(1+math.cos(math.pi*progress))
    lr_scheduler=LambdaLR(optimizer,lambda_para)
    loss=CrossEntropyLoss(ignore_index=model.pad_token_id)
    loss_fn=loss.to(device)
    skip=False
    none_improve=0
    for i in tqdm(range(1,epochs+1)):
        if skip:
            break
        print(f'===== epoch {i} onset =====')
        step=0
        for batch in train_set:
            model.train()
            input,target=batch[:,:-1],batch[:,1:]
            input,target=input.to(device),target.to(device)
            output=model(input)
            curr_loss=loss_fn(output.view(-1,len(vocab)),target.view(-1))
            optimizer.zero_grad()
            curr_loss.backward()
            clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            step+=1
            accumulate_step+=1
            if step % ten_precent==0:
                for name,param in model.named_parameters():
                    if param.grad is not None and torch.isfinite(param.grad).all():
                        runtime_writer.add_histogram(f'grad/{name}',param.grad,accumulate_step)
                runtime_writer.flush()
                model.eval()
                total_loss=0
                with torch.no_grad():
                    for batch in train_set:
                        input,target=batch[:,:-1],batch[:,1:]
                        input,target=input.to(device),target.to(device)
                        output=model(input)
                        valid_set_loss=loss_fn(output.view(-1,len(vocab)),target.view(-1))
                        total_loss+=valid_set_loss.item()
                average_loss=total_loss/len(train_set)
                average_ppl=math.exp(average_loss)
                print(f'\nStep {step} finished, average loss:{curr_loss},average_ppl:{average_ppl} \n',flush=True)
                runtime_writer.add_scalar('ppl',average_ppl,accumulate_step)
                runtime_writer.flush()
                if average_ppl<best_ppl*(1-args.relative_rate):
                    best_ppl=average_ppl
                    none_improve=0
                    torch.save(model.state_dict(),f'ckpt/seed_{current_seed}_{dataset}.pt')
                    # print(f'\n seed {current_seed} epoch {i} update best ppl')
                else:
                    none_improve+=1
                    if none_improve>=args.patience:
                        skip=True
                        shutil.copyfile(f'ckpt/seed_{current_seed}_{dataset}.pt',f'ckpt/best.pt')
                        break

    runtime_writer.add_hparams(hparams,{'ppl':best_ppl},run_name=f'seed_{current_seed}')
    runtime_writer.flush()
    runtime_writer.close()

        # del model                    # 删除模型
        # torch.cuda.empty_cache()     # 清空缓存
        # gc.collect() 
            # if args.seed!=114514:
            #     break
    with open(f'ckpt/dimention_parameter.json','w')as f:
        json.dump(hparams,f, indent=2, ensure_ascii=False, sort_keys=True)
    d, dff, V = args.dimention_of_model, args.dimention_of_FFN, len(vocab)
    attn = 4 * (d*d + d)              # W_Q,W_K,W_V,W_O 四个 Linear(d,d)
    ffn  = (d*dff + dff) + (dff*d + d)   # FFN 两个 Linear
    ln   = 2 * (2*d)                  # norm1, norm2
    per  = attn + ffn + ln             # 每层
    fixed = V*d + (d*V + V)            # 嵌入 + lm_head（含 bias）
    formula = fixed + args.layer_number * per

    print(f'参数配置: d_model={d} n_head={args.number_head} d_ff={dff} N={args.layer_number} V={V}')
    print(f'  每层: 注意力 {attn:,} + FFN {ffn:,} + LayerNorm {ln:,} = {per:,}')
    print(f'  固定: 嵌入 {V*d:,} + lm_head {d*V+V:,} = {fixed:,}')
    print(f'  公式     = {formula:,}')
    print(f'all done, total time comsumption: {time.time()-start_time} s')
if __name__=='__main__':
    main()