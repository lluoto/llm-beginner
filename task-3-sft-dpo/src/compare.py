import argparse
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,Dataset
from src.chat import format_messages,build_labels
import os 
from torch.nn.utils.rnn import pad_sequence
from src.lora import inject_lora
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch.nn.functional as F
from pathlib import Path
import json
from tqdm import tqdm
from time import time
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

runtime_writer=SummaryWriter(f'runs/sample/')

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build_dpo_batch(file_dir,model,max_len=512,pad_token_id=0):
    with open(file_dir,'r') as f:
        a=json.load(f)


    max_len=int(max_len)
    all_data=[]
    for sessions in tqdm(a,desc='building dpo labels'):
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
    parser.add_argument('--dataset','-d',type=str,default='data/dpo_mixed_20k.json')
    parser.add_argument('--train_rate','-t',type=float,default=0.03)
    parser.add_argument('--batch_size','-b',type=int,default='2')
    parser.add_argument('--seed','-s',type=int,default='114514')
    parser.add_argument('--relative_rate','-rr',type=float,default='0.05')
    parser.add_argument('--learning_rate','-lr',type=float,default='4e-3')
    parser.add_argument('--r_lora_rate','-r',type=int,default='8')
    parser.add_argument('--alpha_lora_rate','-a',type=int,default='16')
    parser.add_argument('--max_len','-m',type=int,default='512')
    parser.add_argument('--new_token ','-n',type=int,default='128')
    parser.add_argument('--dpo_beta ','-beta',type=int,default=0.1)
    parser.add_argument('--weight_decay','-wd',type=float,default=0.08)
    return parser



def main():
    args=parser_para().parse_args()
    start_time=time.time()
    
    class DPODastaset(Dataset):
        def __init__(self,data):
            self.data=data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    row_data=build_dpo_batch(args.dataset,tok,tok.model_max_length)
    built_data=DPODastaset(row_data)
    
    valid_set=DataLoader(dataset=built_data,batch_size=args.batch_size,shuffle=True)
    for i in range(3):
        ref=model.requires_grad_(False).eval()
        accuracy_list=[]
        total_loss=0
        if i==0:
            print('apo')
            model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,   # 或 "auto"
            device_map="auto",           # 或指定 device
            low_cpu_mem_usage=True,
            ).to(device).eval()
        elif i==1:
            model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,   # 或 "auto"
            device_map="auto",           # 或指定 device
            low_cpu_mem_usage=True,
            ).to(device).eval()
            model=inject_lora(model,target_modules=['q_proj','k_proj','v_proj'],r=8,alpha=16)
            state=torch.load('ckpt/sft/best_sft.pt')
            model.load_state_dict(state,strict=False)
        elif i==2:
            model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,   # 或 "auto"
            device_map="auto",           # 或指定 device
            low_cpu_mem_usage=True,
            ).to(device).eval()
            model=inject_lora(model,target_modules=['q_proj','k_proj','v_proj'],r=8,alpha=16)
            state=torch.load('ckpt/dpo/best_dpo.pt')
            model.load_state_dict(state,strict=False)
        with torch.no_grad():
            for batch in valid_set:
                model.eval()
                win_id,win_label,lose_id,lose_label=batch.values()
                output_win_id=model.generate(win_id)
                accuracy_list.appenc(torch.argmax())
        t = time() - start_time
        print("time consumption =", t, type(t))
        runtime_writer.add_hparams(accuracy_list,{'timeconsumption':t})
if __name__=='__main__':
    main()