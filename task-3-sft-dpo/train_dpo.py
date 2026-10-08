import copy
import shutil
from torch.nn.modules.loss import CrossEntropyLoss
import argparse
import torch
import random
from transformers import AutoTokenizer,AutoModelForCausalLM
from pathlib import Path
import numpy as np
import json
import torch.nn.functional as F
from src.chat import format_messages,build_labels
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader,Subset,Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW
from tqdm import tqdm
import time
import math
import os 
os.makedirs('ckpt/dpo/',exist_ok=True)
ROOT = Path(__file__).resolve().parents[0]
model_path = ROOT / "models" / "Qwen2.5-0.5B"
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


def logprob_of_sequence(model,prompt,response,mask=None):
    if len(prompt)+len(response)-1 > model.block_size:
        raise ValueError('prefer sample surpass max len')
    x,y = build_dpo_batch([(prompt,response)],max_len=model.block_size)
    logit,_=model(x.to(device),mask)
    token_logps=F.log_softmax(logit,dim=-1).gather(-1,y.to(device).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return token_logps.masked_select(y!=-100).sum()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def collate_pad_crop(batch):
    input_ids=[item['input_ids'] for item in batch]
    input_labels=[item['labels'] for item in batch]
    input_ids_padded=pad_sequence(input_ids,batch_first=True,padding_value=0)
    input_labels_padded=pad_sequence(input_labels,batch_first=True,padding_value=-100)
    return {'input_ids':input_ids_padded,'labels':input_labels_padded}

def build_dpo_batch(file_dir,model,max_len=512,pad_token_id=0):
    with open('data/dpo/dpo_en.json','r') as f:
        a=json.load(f)
        # print(a[:10][1]['conversations'])#conversations，chosen,rejected

    with open(file_dir.replace('zh','zh_simple'),'a',encoding='utf-8')as f:
        cut=a[:300]
        json.dumps(cut,ensure_ascii=False,indent=2)
    max_len=int(max_len)
    all_data=[]
    for sessions in tqdm(a,desc='building dpo labels'):
        win_sessions_list,loss_sessions_list=[],[]
        win_sessions_list.append({'role':'user','content':sessions['conversations']})
        win_sessions_list.append({'role':'assistant','content':sessions['chosen']})
        win_input_id=model(format_messages(win_sessions_list), return_tensors="pt").input_ids[0]
        win_input_label=build_labels(win_input_id,win_sessions_list)
        loss_sessions_list.append({'role':'user','content':sessions['conversations']})
        loss_sessions_list.append({'role':'assistant','content':sessions['chosen']})
        loss_input_id=model(format_messages(loss_sessions_list), return_tensors="pt").input_ids[0]
        loss_input_label=build_labels(win_input_id,loss_sessions_list)
        all_data.append({'input_ids':{'win':win_input_id,'loss':loss_input_id},
                        'labels':{'win':win_input_label,'loss':loss_input_label}})
    return all_data


def parser_para():
    parser=argparse.ArgumentParser(description='Transformer for emotion labeling')
    parser.add_argument('--dataset','-d',type=str,default='data/dpo/dpo_zh.json')
    parser.add_argument('--train_rate','-t',type=float,default=0.03)
    parser.add_argument('--epoches','-e',type=int,default='50')
    parser.add_argument('--batch_size','-b',type=int,default='2')
    parser.add_argument('--seed','-s',type=int,default='114514')
    parser.add_argument('--relative_rate','-rr',type=float,default='0.05')
    parser.add_argument('--learning_rate','-lr',type=float,default='4e-3')
    parser.add_argument('--max_len','-max',type=int,default=4096)
    parser.add_argument('--r_lora_rate','-r',type=int,default='8')
    parser.add_argument('--alpha_lora_rate','-a',type=int,default='16')
    parser.add_argument('--weight_decay','-wd',type=float,default=0.08)
    return parser


def main():
    args=parser_para().parse_args()
    epoches=args.epoches
    tokenizer= AutoTokenizer.from_pretrained(str(model_path))
    start_time=time.time()
    dpo_generator=torch.Generator().manual_seed(current_seed)


    accumulate_step=0
    current_seed=args.seed
    if args.seed==114514:
        current_seed=random.randint(0,1000002)
    set_seed(current_seed)

    hparams={
            'learning_rate':args.learning_rate,
            'train_rate':args.train_rate,
            'max_len':args.max_len,
            'batch_size':args.batch_size,
            'weight_decay':args.weight_decay,
            'alpha_lora_rate':args.alpha_lora_rate,
            'r_lora_rate':args.r_lora_rate,
            
            }
    runtime_writer=SummaryWriter(f'runs/')
    
    class DPODastaset(Dataset):
        def __init__(self,data):
            self.data=data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    row_data=build_dpo_batch(args.dataset,tokenizer,args.max_len)
    built_data=DPODastaset(row_data)
    all_indice=torch.randperm(len(built_data),generator=dpo_generator)
    edge=len(all_indice)//10*9
    train_indice=all_indice[:edge]
    test_indice=all_indice[edge:]
    train_set=DataLoader(dataset=Subset(built_data,train_indice),batch_size=args.batch_size,collate_fn=collate_pad_crop,shuffle=True)
    test_set=DataLoader(dataset=Subset(built_data,test_indice),batch_size=args.batch_size,collate_fn=collate_pad_crop,shuffle=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,   # 或 "auto"
        device_map="auto",           # 或指定 device
        low_cpu_mem_usage=True,
        )
    policy=copy.deepcopy(model).eval()
    ref=copy.deepcopy(model).requires_grad_(False).eval()
    
    optimizer=AdamW([p for p in policy.parameters() if p.requires_grad],args.learning_rate,weight_decay=args.weight_decay)
    skip=False
    none_improve=0
    # casual_mask=torch.triu(torch.ones(L,L,device=x.device),diagonal=1).bool()

    for i in tqdm(range(1,epoches+1)):
        if skip:
            break
        print(f'===== epoch {i} onset =====')
        step=0
        for batch in train_set:
            input,target=batch['input_ids'].to(device),batch['labels'].to(device)
            lp_w=logprob_of_sequence(policy,input['win'],target['win'])
            lp_l=logprob_of_sequence(policy,input['loss'],target['loss'])
            with torch.no_grad():
                lr_w=logprob_of_sequence(ref,input['win'],target['win'])
                lr_l=logprob_of_sequence(ref,input['loss'],target['loss'])
            curr_loss= -F.logsimoid(0.1*(lp_w-lr_w)-(lp_l-lr_l))

            optimizer.zero_grad()
            curr_loss.backward()
            clip_grad_norm_(policy.parameters(),max_norm=1.0)
            optimizer.step()
            step+=1
            accumulate_step+=1
            best_ppl=float('inf')
            if step % 20==0:
                policy.eval()
                total_loss=0
                with torch.no_grad():
                    for test in test_set:
                        test_input,test_target=test['input_ids'].to(device),test['labels'].to(device)
                        output=policy(input_ids=test_input,labels=test_target)
                        valid_set_loss=output.loss
                        total_loss+=valid_set_loss.item()
                average_loss=total_loss/len(train_set)
                average_ppl=math.exp(average_loss)
                print(f'\nStep {step} finished, average loss:{curr_loss},average_ppl:{average_ppl} \n',flush=True)
                runtime_writer.add_scalar('ppl',average_ppl,accumulate_step)
                runtime_writer.flush()
                if average_ppl<best_ppl*(1-args.relative_rate):
                    best_ppl=average_ppl
                    none_improve=0
                    torch.save(policy.state_dict(),f'ckpt/dpo/{i}.pt')
                    print(f'\n seed {current_seed} epoch {i} update best ppl')
                else:
                    none_improve+=1
                    if none_improve>=args.patience:
                        skip=True
                        shutil.copyfile(f'ckpt','ckpt/dpo/best.pt')
                        break

        runtime_writer.add_hparams(hparams,{'ppl':best_ppl},run_name=f'seed_{current_seed}')
        runtime_writer.flush()
        runtime_writer.close()

        # del model                    # 删除模型
        # torch.cuda.empty_cache()     # 清空缓存
        # gc.collect() 
        if args.seed!=114514:
            break

    print(f'all done, total time comsumption: {time.time()-start_time} s')
if __name__=='__main__':
    main()