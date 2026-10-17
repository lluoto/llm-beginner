import argparse
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,Dataset
from src.chat import format_messages,build_labels
import os 
from src.lora import inject_lora
from transformers import AutoTokenizer,AutoModelForCausalLM
from pathlib import Path
import json
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]
model_path = ROOT / "models" / "Qwen2.5-0.5B"
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
def collate_pad(batch):
    win_input_ids=[item['input_ids']['win'] for item in batch]
    win_input_labels=[item['labels']['win'] for item in batch]
    lose_input_ids=[item['input_ids']['loss'] for item in batch]
    lose_input_labels=[item['labels']['loss'] for item in batch]
    win_prompt_ids=[item['input_ids']['prompt']for item in batch]
    win_input_ids_padded=pad_sequence(win_input_ids,batch_first=True,padding_value=151643)
    win_prompt_ids_padded=pad_sequence(win_prompt_ids,batch_first=True,padding_value=151643)
    win_input_labels_padded=pad_sequence(win_input_labels,batch_first=True,padding_value=-100)
    lose_input_ids_padded=pad_sequence(lose_input_ids,batch_first=True,padding_value=151643)
    lose_input_labels_padded=pad_sequence(lose_input_labels,batch_first=True,padding_value=-100)
    win_attn=(win_input_ids_padded!=151643).long()
    lose_attn=(lose_input_ids_padded!=151643).long()
    return {'win_input_ids':win_input_ids_padded,'win_labels':win_input_labels_padded,'win_attn':win_attn,
            'lose_input_ids':lose_input_ids_padded,'lose_labels':lose_input_labels_padded,'lose_attn':lose_attn,
            'win_prompt_ids':win_prompt_ids_padded
            }
def stats(gen_texts):
    avg_len=sum(len(t) for t in gen_texts)/len(gen_texts)
    def rep(t):
        g=[t[i:i+3] for i in range(len(t)-2)]
        return 1-len(set(g))/max(len(g),1)
    avg_rep=sum(rep(t) for t in gen_texts)/len(gen_texts)
    return avg_len,avg_rep
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build_batch(file_dir,model,max_len=512,pad_token_id=0):
    with open(file_dir,'r') as f:
        a=json.load(f)
    max_len=int(max_len)
    all_data=[]
    for sessions in tqdm(a,desc='building dpo labels'):
        win_sessions_list,lose_sessions_list=[],[]
        win_sessions_list.append({'role':'user','content':sessions['conversations'][0]['value']})
        win_prompt_id=model(format_messages(win_sessions_list), return_tensors="pt").input_ids[0]
        win_sessions_list.append({'role':'assistant','content':sessions['chosen']['value']})
        win_input_id=model(format_messages(win_sessions_list), return_tensors="pt").input_ids[0]
        lose_sessions_list.append({'role':'user','content':sessions['conversations'][0]['value']})
        lose_sessions_list.append({'role':'assistant','content':sessions['rejected']['value']})
        lose_input_id=model(format_messages(lose_sessions_list), return_tensors="pt").input_ids[0]
        if win_input_id.shape[-1]>max_len or lose_input_id.shape[-1]>max_len:
            continue
        lose_input_label=build_labels(lose_input_id,lose_sessions_list)
        win_input_label=build_labels(win_input_id,win_sessions_list)

        all_data.append({'input_ids':{'win':win_input_id,'loss':lose_input_id,'prompt':win_prompt_id},
                        'labels':{'win':win_input_label,'loss':lose_input_label}})
    return all_data
class_model={0:'apo',1:'sft',2:'dpo'}
def parser_para():
    parser=argparse.ArgumentParser(description='Transformer for emotion labeling')
    parser.add_argument('--dataset','-d',type=str,default='data/dpo_mixed_20k.json')
    parser.add_argument('--batch_size','-b',type=int,default='2')
    parser.add_argument('--seed','-s',type=int,default='114514')
    parser.add_argument('--r_lora_rate','-r',type=int,default='8')
    parser.add_argument('--alpha_lora_rate','-a',type=int,default='16')
    parser.add_argument('--max_len','-m',type=int,default='512')
    parser.add_argument('--new_token','-n',type=int,default='128')
    parser.add_argument('--dpo_beta','-beta',type=int,default=0.1)
    return parser

        
def main():
    args=parser_para().parse_args()
    
    class DPODastaset(Dataset):
        def __init__(self,data):
            self.data=data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    built_data=DPODastaset(build_batch(args.dataset,tok,tok.model_max_length))
    
    valid_set=DataLoader(dataset=built_data,batch_size=args.batch_size,collate_fn=collate_pad,shuffle=True)
    runtime_writer=SummaryWriter('runs/sample/')
    set_seed(args.seed)
    for i in range(3):
        if i==0:
            model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,   # 或 "auto"
            low_cpu_mem_usage=True,
            ).to(device).eval()
        elif i==1:
            model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,   # 或 "auto"
            low_cpu_mem_usage=True,
            ).to(device).eval()
            model=inject_lora(model,target_modules=['q_proj','k_proj','v_proj'],r=args.r_lora_rate,alpha=args.alpha_lora_rate)
            state=torch.load('ckpt/sft/best_sft.pt')
            model.load_state_dict(state,strict=False)
        elif i==2:
            model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,   # 或 "auto"
            low_cpu_mem_usage=True,
            ).to(device).eval()
            model=inject_lora(model,target_modules=['q_proj','k_proj','v_proj'],r=args.r_lora_rate,alpha=args.alpha_lora_rate)
            state=torch.load('ckpt/dpo/best_dpo.pt')
            model.load_state_dict(state,strict=False)
        with torch.no_grad():
            accumulation_step=1
            accuracy_count=0
            for batch in valid_set:
                model.eval()
                win_ids,win_label,_,lose_ids,lose_label,_,win_prompt_id=batch.values()
                win_ids,win_label,lose_ids,lose_label,win_prompt_id=win_ids.to(device),win_label.to(device),lose_ids.to(device),lose_label.to(device),win_prompt_id.to(device)
                win_lg = model(win_ids).logits[:, :-1]
                lose_lg = model(lose_ids).logits[:, :-1]
                lp_w = (torch.log_softmax(win_lg, -1).gather(-1, win_label[:,1:].unsqueeze(-1)).clamp_min(0).squeeze(-1*(win_label[:,1:]!=-100))).sum(dim=-1)
                lp_l = (torch.log_softmax(lose_lg, -1).gather(-1, lose_label[:,1:].unsqueeze(-1)).clamp_min(0).squeeze(-1*(lose_label[:,1:]!=-100))).sum(dim=-1)
                if lp_w>lp_l:accuracy_count+=1
                output=model.generate(win_prompt_id,max_new_tokens=args.new_token,do_sample=False,pad_token_id=tok.pad_token_id)
                response=tok.decode(output[:,win_prompt_id.shape[1]:],skip_special_tokens=True)
                ave_len,ave_rep=stats(response)
                runtime_writer.add_scalar(f'metric/{class_model[i]}_trend',accuracy_count/len(valid_set),accumulation_step)
                runtime_writer.add_scalar(f'metric/{class_model[i]}_avg_len',ave_len,accumulation_step)
                runtime_writer.add_scalar(f'metric/{class_model[i]}_avg_rep',ave_rep,accumulation_step)
                runtime_writer.add_text(f'text/{class_model[i]}',response,global_step=None,walltime=None)
                accumulation_step+=1
        del model
        torch.cuda.empty_cache()
if __name__=='__main__':
    main()
