import shutil
from torch.nn.modules.loss import CrossEntropyLoss
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
import gc
import math
import os 
os.makedirs('ckpt/sft/',exist_ok=True)
from src.lora import inject_lora
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

def clean_moss(text, tag, end_tag):
    """去掉 <|tag|>: 前缀和结尾标记"""
    text = text.replace(f'<|{tag}|>:', '').strip()
    text = text.replace(end_tag, '').strip()
    return text

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def collate_pad_crop(batch):
    input_ids=[item['input_ids'] for item in batch]
    input_labels=[item['labels'] for item in batch]
    input_ids_padded=pad_sequence(input_ids,batch_first=True,padding_value=0)
    input_labels_padded=pad_sequence(input_labels,batch_first=True,padding_value=-100)
    return {'input_ids':input_ids_padded,'labels':input_labels_padded}

def build_sft_batch(file_dir,model,max_len=512,pad_token_id=0):
    with open(file_dir,'r',encoding='utf-8') as f:
        r=[json.loads(line) for line in f if line.strip()][:500]

    # with open(file_dir.replace('moss-003-sft-no-tools','simple'),'a',encoding='utf-8')as f:
    #     for item in r[:300]:
    #         f.write(json.dumps(item,ensure_ascii=False)+'\n')
    max_len=int(max_len)
    all_data=[]
    for sessions in tqdm(r,desc='building sft labels'):
        sessions_list=[]
        input_ids,input_labels=[],[]
        for turn in sessions['chat']:
            clean_response=clean_moss(sessions['chat'][turn]['MOSS'], '<|MOSS|>:', '<eom>')
            sessions_list.append({'role':'user','content':clean_moss(sessions['chat'][turn]['Human'], '<|Human|>:', '<eoh>')})
            sessions_list.append({'role':'assistant','content':clean_response})
        input_id=model(format_messages(sessions_list), return_tensors="pt").input_ids[0]
        input_label=build_labels(input_id,sessions_list)

        all_data.append({'input_ids':input_id,
                        'labels':input_label})
    return all_data


def parser_para():
    parser=argparse.ArgumentParser(description='Transformer for emotion labeling')
    # parser.add_argument('--dataset','-d',type=str,default='data/moss-sft/moss-003-sft-no-tools.jsonl')
    parser.add_argument('--dataset','-d',type=str,default='data/moss-sft/simple.jsonl')
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


    accumulate_step=0
    current_seed=args.seed
    if args.seed==114514:
        current_seed=random.randint(0,1000002)
    set_seed(current_seed)
    sft_generator=torch.Generator().manual_seed(current_seed)

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
    
    class SFTDastaset(Dataset):
        def __init__(self,data):
            self.data=data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    row_data=build_sft_batch(args.dataset,tokenizer,args.max_len)
    built_data=SFTDastaset(row_data)
    all_indice=torch.randperm(len(built_data),generator=sft_generator)
    edge=len(all_indice)//10*9
    train_indice=all_indice[:edge]
    test_indice=all_indice[edge:]
    train_set=DataLoader(dataset=Subset(built_data,train_indice),batch_size=args.batch_size,collate_fn=collate_pad_crop,shuffle=True)
    test_set=DataLoader(dataset=Subset(built_data,test_indice),batch_size=args.batch_size,collate_fn=collate_pad_crop,shuffle=True)
    model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,   # 或 "auto"
        device_map="auto",           # 或指定 device
        low_cpu_mem_usage=True,
        )
    lora_model=inject_lora(model,target_modules=['q_proj','k_proj','v_proj'],r=args.r_lora_rate,alpha=args.alpha_lora_rate,trainable_rate=args.train_rate)
    
    lora_model=lora_model.to(device)
    lora_model.train()

    optimizer=AdamW(lora_model.parameters(),args.learning_rate,weight_decay=args.weight_decay)
    skip=False
    none_improve=0
    for i in tqdm(range(1,epoches+1)):
        if skip:
            break
        print(f'===== epoch {i} onset =====')
        step=0
        for batch in train_set:
            input,target=batch['input_ids'].to(device),batch['labels'].to(device)
            output=lora_model(input_ids=input,labels=target)
            curr_loss=output.loss
            optimizer.zero_grad()
            curr_loss.backward()
            clip_grad_norm_(lora_model.parameters(),max_norm=1.0)
            optimizer.step()
            step+=1
            accumulate_step+=1
            best_ppl=float('inf')
            if step % 20==0:
                lora_model.eval()
                total_loss=0
                with torch.no_grad():
                    for test in test_set:
                        test_input,test_target=test['input_ids'].to(device),test['labels'].to(device)
                        output=lora_model(input_ids=test_input,labels=test_target)
                        valid_set_loss=output.loss
                        total_loss+=valid_set_loss.item()
                average_loss=total_loss/len(test)
                average_ppl=math.exp(average_loss)
                print(f'\nStep {step} finished, average loss:{curr_loss},average_ppl:{average_ppl} \n',flush=True)
                runtime_writer.add_scalar('ppl',average_ppl,accumulate_step)
                runtime_writer.flush()
                if average_ppl<best_ppl*(1-args.relative_rate):
                    best_ppl=average_ppl
                    none_improve=0
                    lora_state = {
                        k: v for k, v in model.state_dict().items()
                        if 'lora' in k.lower()          # 按你的命名规则
                    }
                    torch.save(lora_state,f'ckpt/sft/{i}.pt')
                    print(f'\n seed {current_seed} epoch {i} update best ppl')
                else:
                    none_improve+=1
                    if none_improve>=args.patience:
                        skip=True
                        shutil.copyfile(f'ckpt','ckpt/sft/best.pt')
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