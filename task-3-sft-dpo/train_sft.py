import shutil
import os 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
import torch
import random
from transformers import AutoTokenizer,AutoModelForCausalLM
from pathlib import Path
import numpy as np
import json
from src.chat import format_messages,build_labels
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader,Subset,Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.adamw import AdamW
from tqdm import tqdm
import time
from itertools import islice
import math
os.makedirs('ckpt/sft/',exist_ok=True)
from src.lora import inject_lora,merge_lora
ROOT = Path(__file__).resolve().parents[0]
model_path = ROOT / "models" / "Qwen2.5-0.5B"
def pad_mask_generate(input_tensor,pad_token_id=151643):
    
    return (input_tensor != pad_token_id).long()    # 非 padding 位置为 True
def set_seed(seed=1111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def clean_moss(text, tag, end_tag):
    """去掉 <|tag|>: 前缀和结尾标记"""
    text = text.replace(f'<|{tag}|>:', '').strip()
    text = text.replace(end_tag, '').strip()
    return text

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def collate_pad_crop(batch):
    input_ids=[item['input_ids'] for item in batch]
    input_labels=[item['labels'] for item in batch]
    input_ids_padded=pad_sequence(input_ids,batch_first=True,padding_value=151643)
    attention_mask=pad_mask_generate(input_ids_padded)
    input_labels_padded=pad_sequence(input_labels,batch_first=True,padding_value=-100)
    return {'input_ids':input_ids_padded,'labels':input_labels_padded,'attention_mask':attention_mask}

def build_sft_batch(file_dir,model,max_len=512,pad_token_id=151643
                    ):
    with open(file_dir,'r',encoding='utf-8') as f:
        r=[json.loads(line) for line in islice((l for l in f if l.strip()),20000)]

    # with open(file_dir.replace('moss-003-sft-no-tools','simple'),'a',encoding='utf-8')as f:
    #     for item in r[:300]:
    #         f.write(json.dumps(item,ensure_ascii=False)+'\n')
    max_len=int(max_len)
    all_data=[]
    for sessions in tqdm(r,desc='building sft labels'):
        sessions_list=[]
        input_ids,input_labels=[],[]
        for turn in sessions['chat']:
            clean_response=clean_moss(sessions['chat'][turn]['MOSS'], 'MOSS', '<eom>')
            sessions_list.append({'role':'user','content':clean_moss(sessions['chat'][turn]['Human'], 'Human', '<eoh>')})
            sessions_list.append({'role':'assistant','content':clean_response})
        input_id=model(format_messages(sessions_list), truncation=True,max_length=int(max_len),return_tensors="pt").input_ids[0]
        input_label=build_labels(input_id,sessions_list)

        all_data.append({'input_ids':input_id,
                        'labels':input_label})
    return all_data


def parser_para():
    parser=argparse.ArgumentParser(description='Transformer for emotion labeling')
    # parser.add_argument('--dataset','-d',type=str,default='data/moss-003-sft-no-tools.jsonl')
    parser.add_argument('--dataset','-d',type=str,default='data/simple.jsonl')
    parser.add_argument('--epoches','-e',type=int,default='50')
    parser.add_argument('--batch_size','-b',type=int,default='2')
    parser.add_argument('--seed','-s',type=int,default='114514')
    parser.add_argument('--relative_rate','-rr',type=float,default='0.05')
    parser.add_argument('--learning_rate','-lr',type=float,default='2e-4')
    parser.add_argument('--max_len','-max',type=int,default=4096)
    parser.add_argument('--r_lora_rate','-r',type=int,default='8')
    parser.add_argument('--alpha_lora_rate','-a',type=int,default='16')
    parser.add_argument('--patience','-p',type=int,default=5)
    parser.add_argument('--weight_decay','-wd',type=float,default=0.08)
    return parser


def main():
    args=parser_para().parse_args()
    epoches=args.epoches
    tokenizer= AutoTokenizer.from_pretrained(str(model_path))
    start_time=time.time()


    accumulate_step=0
    current_seed=args.seed
    if args.seed==114514:
        current_seed=random.randint(0,1000002)
    set_seed(current_seed)
    sft_generator=torch.Generator().manual_seed(current_seed)

    hparams={
            'learning_rate':args.learning_rate,
            'max_len':args.max_len,
            'batch_size':args.batch_size,
            'weight_decay':args.weight_decay,
            'alpha_lora_rate':args.alpha_lora_rate,
            'r_lora_rate':args.r_lora_rate,
            
            }
    runtime_writer=SummaryWriter('runs/')
    
    class SFTDataset(Dataset):
        def __init__(self,data):
            self.data=data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    row_data=build_sft_batch(args.dataset,tokenizer,args.max_len)
    built_data=SFTDataset(row_data)
    all_indice=torch.randperm(len(built_data),generator=sft_generator)
    edge=len(all_indice)//10*9
    train_indice=all_indice[:edge]
    test_indice=all_indice[edge:]
    train_set=DataLoader(dataset=Subset(built_data,train_indice),batch_size=args.batch_size,collate_fn=collate_pad_crop,shuffle=True)
    test_set=DataLoader(dataset=Subset(built_data,test_indice),batch_size=args.batch_size,collate_fn=collate_pad_crop,shuffle=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,   # 或 "auto"
        low_cpu_mem_usage=True,
        ).to(device)
    lora_model=inject_lora(model,target_modules=['q_proj','k_proj','v_proj'],r=args.r_lora_rate,alpha=args.alpha_lora_rate)
    
    lora_model=lora_model.to(device)
    lora_model.train()
    best_ppl=float('inf')

    optimizer=AdamW(lora_model.parameters(),args.learning_rate,weight_decay=args.weight_decay)
    skip=False
    none_improve=0
    for i in tqdm(range(1,epoches+1)):
        if skip:
            break
        print(f'===== epoch {i} onset =====')
        step=0
        for batch in train_set:
            lora_model.train()
            input,target,attn_mask=batch['input_ids'].to(device),batch['labels'].to(device),batch['attention_mask'].to(device)
            output=lora_model(input_ids=input,labels=target,attention_mask=attn_mask)
            curr_loss=output.loss
            optimizer.zero_grad()
            curr_loss.backward()
            clip_grad_norm_(lora_model.parameters(),max_norm=1.0)
            optimizer.step()
            step+=1
            accumulate_step+=1
            if step % 20==0:
                lora_model.eval()
                total_loss=0
                with torch.no_grad():
                    for test in test_set:
                        test_input,test_target,test_mask=test['input_ids'].to(device),test['labels'].to(device),test['attention_mask'].to(device)
                        output=model(input_ids=test_input,labels=test_target,attention_mask=test_mask)
                        valid_set_loss=output.loss
                        total_loss+=valid_set_loss.item()
                average_loss=total_loss/len(test_set)
                average_ppl=math.exp(average_loss)
                print(f'\nStep {step} finished, average train loss:{curr_loss},average_ppl:{average_ppl} \n',flush=True)
                runtime_writer.add_scalar('ppl',average_ppl,accumulate_step)
                runtime_writer.flush()
                if average_ppl<best_ppl*(1-args.relative_rate):
                    best_ppl=average_ppl
                    none_improve=0
                    lora_state = {
                        k: v for k, v in model.state_dict().items()
                        if 'lora' in k.lower()          # 按你的命名规则
                    }
                    best_file=f'ckpt/sft/{i}.pt'

                    torch.save(lora_state,f'ckpt/sft/{i}.pt')
                    print(f'\n seed {current_seed} epoch {i} update best ppl')
                else:
                    none_improve+=1
                    if none_improve>=args.patience:
                        skip=True
                        shutil.copyfile(best_file,'ckpt/sft/best.pt')
                        break

    runtime_writer.add_hparams(hparams,{'ppl':best_ppl},run_name=f'seed_{current_seed}')
    runtime_writer.flush()
    runtime_writer.close()

        # del model                    # 删除模型
        # torch.cuda.empty_cache()     # 清空缓存
        # gc.collect() 


    print(f'all done, total time comsumption: {time.time()-start_time} s')
if __name__=='__main__':
    main()