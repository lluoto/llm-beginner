from src.lora import inject_lora
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
import torch
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
ROOT = Path(__file__).resolve().parents[1]
tok = AutoTokenizer.from_pretrained(str(model_path))
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


def logprob_of_sequence(model,prompt,response,mask=None,max_len=512):
    if len(prompt)+len(response)-1 > max_len:
        raise ValueError('prefer sample surpass max len')
    out=model(prompt.to(device),mask=mask)
    logit=out.logits
    response=response.to(device)
    token_logps=F.log_softmax(logit,dim=-1).gather(-1,response.to(device).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return token_logps.masked_select(response!=-100).sum()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def collate_pad(batch):
    win_input_ids=[item['input_ids']['win'] for item in batch]
    win_input_labels=[item['labels']['win'] for item in batch]
    lose_input_ids=[item['input_ids']['loss'] for item in batch]
    lose_input_labels=[item['labels']['loss'] for item in batch]
    win_input_ids_padded=pad_sequence(win_input_ids,batch_first=True,padding_value=0)
    win_input_labels_padded=pad_sequence(win_input_labels,batch_first=True,padding_value=-100)
    lose_input_ids_padded=pad_sequence(lose_input_ids,batch_first=True,padding_value=0)
    lose_input_labels_padded=pad_sequence(lose_input_labels,batch_first=True,padding_value=-100)
    win_attn=(win_input_ids_padded!=0).long()
    lose_attn=(lose_input_ids_padded!=0).long()
    return {'win_input_ids':win_input_ids_padded,'win_labels':win_input_labels_padded,'win_attn':win_attn,
            'lose_input_ids':lose_input_ids_padded,'lose_labels':lose_input_labels_padded,'lose_attn':lose_attn
            }

def build_dpo_batch(file_dir,model,max_len=512,pad_token_id=0):
    file_datas=[]
    for file in os.listdir(file_dir):
        if file.endswith('.json'):
            with open(os.path.join(file_dir,file),'r') as f:
                a=json.load(f)
                file_datas+=a
        # print(a[:10][1]['conversations'])#conversations，chosen,rejected

    # with open(file_dir.replace('zh','zh_simple'),'w',encoding='utf-8')as f:
    #     cut=a[:10]
    #     json.dump(cut,f,ensure_ascii=False,indent=2)
    max_len=int(max_len)
    all_data=[]
    for sessions in tqdm(file_datas,desc='building dpo labels'):
        win_sessions_list,lose_sessions_list=[],[]
        win_sessions_list.append({'role':'user','content':sessions['conversations']})
        win_sessions_list.append({'role':'assistant','content':sessions['chosen']['value']})
        win_input_id=model(format_messages(win_sessions_list), return_tensors="pt").input_ids[0]
        win_input_label=build_labels(win_input_id,win_sessions_list)
        lose_sessions_list.append({'role':'user','content':sessions['conversations']})
        lose_sessions_list.append({'role':'assistant','content':sessions['rejected']['value']})
        lose_input_id=model(format_messages(lose_sessions_list), return_tensors="pt").input_ids[0]
        lose_input_label=build_labels(lose_input_id,lose_sessions_list)
        all_data.append({'input_ids':{'win':win_input_id,'loss':lose_input_id},
                        'labels':{'win':win_input_label,'loss':lose_input_label}})
    return all_data


def parser_para():
    parser=argparse.ArgumentParser(description='Transformer for emotion labeling')
    parser.add_argument('--dataset','-d',type=str,default='data/dpo/')
    # parser.add_argument('--dataset','-d',type=str,default='data/dpo/dpo_zh_simple.json')
    parser.add_argument('--train_rate','-t',type=float,default=0.03)
    parser.add_argument('--epoches','-e',type=int,default='50')
    parser.add_argument('--batch_size','-b',type=int,default='2')
    parser.add_argument('--seed','-s',type=int,default='114514')
    parser.add_argument('--relative_rate','-rr',type=float,default='0.05')
    parser.add_argument('--learning_rate','-lr',type=float,default='4e-3')
    parser.add_argument('--r_lora_rate','-r',type=int,default='8')
    parser.add_argument('--alpha_lora_rate','-a',type=int,default='16')
    parser.add_argument('--weight_decay','-wd',type=float,default=0.08)
    return parser


def main():
    args=parser_para().parse_args()
    epoches=args.epoches
    start_time=time.time()


    accumulate_step=0
    current_seed=args.seed
    if args.seed==114514:
        current_seed=random.randint(0,1000002)
    set_seed(current_seed)
    dpo_generator=torch.Generator().manual_seed(current_seed)

    hparams={
            'learning_rate':args.learning_rate,
            'train_rate':args.train_rate,
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
    row_data=build_dpo_batch(args.dataset,tok,tok.model_max_length)
    built_data=DPODastaset(row_data)
    all_indice=torch.randperm(len(built_data),generator=dpo_generator)
    edge=len(all_indice)//10*9
    train_indice=all_indice[:edge]
    test_indice=all_indice[edge:]
    train_set=DataLoader(dataset=Subset(built_data,train_indice),batch_size=args.batch_size,collate_fn=collate_pad,shuffle=True)
    test_set=DataLoader(dataset=Subset(built_data,test_indice),batch_size=args.batch_size,collate_fn=collate_pad,shuffle=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,   # 或 "auto"
        device_map="auto",           # 或指定 device
        low_cpu_mem_usage=True,
        )
    model=inject_lora(model,target_modules=['q_proj','k_proj','v_proj'],r=8,alpha=16)
    state=torch.load('ckpt/sft/10.pt')
    model.load_state_dict(state,strict=False)
    policy=model.train()
    model2 = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,   # 或 "auto"
            device_map="auto",           # 或指定 device
            low_cpu_mem_usage=True,
            )
    model2=inject_lora(model2,target_modules=['q_proj','k_proj','v_proj'],r=8,alpha=16)
    model2.load_state_dict(state,strict=False)
    ref=model2.requires_grad_(False).eval()
    
    optimizer=AdamW([p for p in policy.parameters() if p.requires_grad],args.learning_rate,weight_decay=args.weight_decay)
    skip=False
    none_improve=0

    for i in tqdm(range(1,epoches+1)):
        if skip:
            break
        print(f'===== epoch {i} onset =====')
        step=0
        for batch in train_set:
            win_id,win_label,win_attn,lose_id,lose_label,lose_attn=batch.values()
            lp_w=logprob_of_sequence(policy,win_id,win_label,win_attn)
            lp_l=logprob_of_sequence(policy,lose_id,lose_label,lose_attn)
            with torch.no_grad():
                lr_w=logprob_of_sequence(ref,win_id,win_label,win_attn)
                lr_l=logprob_of_sequence(ref,lose_id,lose_label,lose_attn)
            curr_loss= -F.logsigmoid(0.1*((lp_w-lr_w)-(lp_l-lr_l)))

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
                        win_id,win_label,win_attn,lose_id,lose_label,lose_attn=batch.values()
                        lp_w=logprob_of_sequence(policy,win_id,win_label,win_attn)
                        lp_l=logprob_of_sequence(policy,lose_id,lose_label,lose_attn)
                        with torch.no_grad():
                            lr_w=logprob_of_sequence(ref,win_id,win_label,win_attn)
                            lr_l=logprob_of_sequence(ref,lose_id,lose_label,lose_attn)
                        curr_loss= -F.logsigmoid(0.1*((lp_w-lr_w)-(lp_l-lr_l))).item()
                        total_loss+=curr_loss
                average_loss=total_loss/len(test_set)
                average_ppl=math.exp(average_loss)
                accuracy=((lp_w-lr_w)>(lp_l-lr_l)).float().mean()
                print(f'\nStep {step} finished, average loss:{curr_loss},average_ppl:{average_ppl}, accuracy:{accuracy}\n',flush=True)
                runtime_writer.add_scalar('ppl',average_ppl,accumulate_step)
                runtime_writer.flush()
                if average_ppl<best_ppl*(1-args.relative_rate):
                    best_ppl=average_ppl
                    none_improve=0
                    lora_state={k:v for k,v in policy.state_dict().items() if 'lora' in k.lower()}
                    torch.save(lora_state,f'ckpt/dpo/{i}.pt')
                    print(f'\n seed {current_seed} epoch {i} update best ppl')
                else:
                    none_improve+=1
                    if none_improve>=args.patience:
                        skip=True
                        shutil.copyfile(f'ckpt/dpo/{i}.pt','ckpt/dpo/best.pt')
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