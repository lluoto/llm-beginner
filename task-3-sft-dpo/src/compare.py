import os 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import torch
import random
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,Dataset
from chat import format_messages,build_labels

from lora import inject_lora
from transformers import AutoTokenizer,AutoModelForCausalLM
from pathlib import Path
import json
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]
model_path = ROOT / "models" / "Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(str(model_path))
pad_token_id=151643
def set_seed(seed=1111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
def left_pad(seqs,pad_id):
    max_len=max(s.shape[0] for s in seqs)
    out=torch.full((len(seqs),max_len),pad_id,dtype=torch.long)
    for row,s in enumerate(seqs):
        out[row,max_len-s.shape[0]:]=s
    return out
def collate_pad(batch):
    win_input_ids=[item['input_ids']['win'] for item in batch]
    win_input_labels=[item['labels']['win'] for item in batch]
    lose_input_ids=[item['input_ids']['loss'] for item in batch]
    lose_input_labels=[item['labels']['loss'] for item in batch]
    win_prompt_ids=[item['input_ids']['prompt']for item in batch]
    win_input_ids_padded=pad_sequence(win_input_ids,batch_first=True,padding_value=pad_token_id)
    win_prompt_ids_padded=left_pad(win_prompt_ids,pad_token_id)
    win_input_labels_padded=pad_sequence(win_input_labels,batch_first=True,padding_value=-100)
    lose_input_ids_padded=pad_sequence(lose_input_ids,batch_first=True,padding_value=pad_token_id)
    lose_input_labels_padded=pad_sequence(lose_input_labels,batch_first=True,padding_value=-100)
    win_attn=(win_input_ids_padded!=pad_token_id).long()
    lose_attn=(lose_input_ids_padded!=pad_token_id).long()
    prompt_attn=(win_prompt_ids_padded!=pad_token_id).long()
    return {'win_input_ids':win_input_ids_padded,'win_labels':win_input_labels_padded,'win_attn':win_attn,
            'lose_input_ids':lose_input_ids_padded,'lose_labels':lose_input_labels_padded,'lose_attn':lose_attn,
            'win_prompt_ids':win_prompt_ids_padded,'prompt_attn':prompt_attn
            }
def stats(gen_texts):
    avg_len=sum(len(t) for t in gen_texts)/max(len(gen_texts),1)
    def rep(t):
        g=[t[i:i+3] for i in range(len(t)-2)]
        return 1-len(set(g))/max(len(g),1)
    avg_rep=sum(rep(t) for t in gen_texts)/max(len(gen_texts),1)
    return avg_len,avg_rep
def esc(s):
    return s.replace('\n','\\n')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build_batch(file_dir,model,max_len=512):
    with open(file_dir,'r',encoding='utf-8') as f:
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
class_model={0:'base',1:'sft',2:'dpo'}
def ckpt_path(kind):
    d=ROOT/'ckpt'/kind
    for name in (f'best_{kind}.pt','best.pt'):
        if (d/name).exists():
            return d/name
    cands=sorted(d.glob('*.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
    if cands:
        return cands[0]
    raise FileNotFoundError(f'no checkpoint under {d}')
def load_model(i,args):
    model=AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        ).to(device).eval()
    if i==0:
        return model
    model=inject_lora(model,target_modules=['q_proj','k_proj','v_proj'],r=args.r_lora_rate,alpha=args.alpha_lora_rate)
    path=ckpt_path(class_model[i])
    print(f'[{class_model[i]}] loading {path}',flush=True)
    model.load_state_dict(torch.load(str(path),map_location=device),strict=False)
    return model
def seq_logprob(lg,label,norm):
    mask=(label[:,1:]!=-100)
    lp=(torch.log_softmax(lg,-1).gather(-1,label[:,1:].clamp_min(0).unsqueeze(-1)).squeeze(-1)*mask).sum(dim=-1)
    if norm=='mean':
        lp=lp/mask.sum(dim=-1).clamp_min(1)
    return lp
def parser_para():
    parser=argparse.ArgumentParser(description='compare base / sft / dpo')
    parser.add_argument('--dataset','-d',type=str,default='data/dpo_mixed_20k.json')
    parser.add_argument('--batch_size','-b',type=int,default='2')
    parser.add_argument('--seed','-s',type=int,default='114514')
    parser.add_argument('--r_lora_rate','-r',type=int,default='8')
    parser.add_argument('--alpha_lora_rate','-a',type=int,default='16')
    parser.add_argument('--max_len','-m',type=int,default='4096')
    parser.add_argument('--new_token','-n',type=int,default='128')
    parser.add_argument('--n_probe','-p',type=int,default='50')
    parser.add_argument('--probe_prompt_len',type=int,default='512')
    parser.add_argument('--lp_norm',type=str,default='mean',choices=['mean','sum'])
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
    data_path=Path(args.dataset)
    if not data_path.exists():
        data_path=ROOT/args.dataset
    built_data=DPODastaset(build_batch(str(data_path),tok,args.max_len))
    set_seed(args.seed)
    rng=random.Random(args.seed)
    cand=[k for k,item in enumerate(built_data) if item['input_ids']['prompt'].shape[0]<=args.probe_prompt_len]
    rng.shuffle(cand)
    idx=cand[:args.n_probe]
    probe=[built_data[k] for k in idx]
    valid_set=DataLoader(dataset=DPODastaset(probe),batch_size=args.batch_size,collate_fn=collate_pad,shuffle=False)
    print(f'probe size={len(probe)} (from {len(built_data)} built, {len(cand)} candidates)',flush=True)
    records={}
    for i in range(3):
        name=class_model[i]
        writer=SummaryWriter(str(ROOT/'runs'/'sample'/name))
        model=load_model(i,args)
        correct,total,texts,step=0,0,[],0
        with torch.no_grad():
            for batch in tqdm(valid_set,desc=name):
                win_ids,win_label,win_attn,lose_ids,lose_label,lose_attn,win_prompt_id,prompt_attn=batch.values()
                win_attn,lose_attn,prompt_attn=win_attn.to(device),lose_attn.to(device),prompt_attn.to(device)
                win_ids,win_label,lose_ids,lose_label,win_prompt_id=win_ids.to(device),win_label.to(device),lose_ids.to(device),lose_label.to(device),win_prompt_id.to(device)
                win_lg = model(win_ids,attention_mask=win_attn).logits[:, :-1]
                lose_lg = model(lose_ids,attention_mask=lose_attn).logits[:, :-1]
                lp_w = seq_logprob(win_lg,win_label,args.lp_norm)
                lp_l = seq_logprob(lose_lg,lose_label,args.lp_norm)
                correct+=int((lp_w>lp_l).sum())
                total+=lp_w.shape[0]
                output=model.generate(win_prompt_id,attention_mask=prompt_attn,max_new_tokens=args.new_token,do_sample=False,pad_token_id=pad_token_id)
                response=[tok.decode(output[j,win_prompt_id.shape[1]:],skip_special_tokens=True) for j in range(output.shape[0])]
                for j,r in enumerate(response):
                    k=step*args.batch_size+j
                    if k<len(probe):
                        prompt_text=tok.decode(probe[k]['input_ids']['prompt'],skip_special_tokens=True)
                        writer.add_text(f'text/{name}',f'[prompt] {prompt_text}\n[response] {r}',global_step=k)
                texts.extend(response)
                writer.add_scalar(f'metric/{name}_acc',correct/max(total,1),step)
                step+=1
        ave_len,ave_rep=stats(texts)
        acc=correct/max(total,1)
        writer.add_scalar(f'metric/{name}_avg_len',ave_len,0)
        writer.add_scalar(f'metric/{name}_avg_rep',ave_rep,0)
        writer.add_scalar(f'metric/{name}_acc_final',acc,0)
        writer.flush()
        writer.close()
        records[name]=texts
        print(f'[{name}] acc={acc:.4f} avg_len={ave_len:.1f} avg_rep={ave_rep:.3f}',flush=True)
        del model
        torch.cuda.empty_cache()
    lines=[]
    for k,item in enumerate(probe):
        prompt_text=tok.decode(item['input_ids']['prompt'],skip_special_tokens=True)
        lines.append(f'===== sample {k} =====')
        lines.append(f'[prompt] {esc(prompt_text)}')
        for name in ('base','sft','dpo'):
            t=records[name][k] if k<len(records[name]) else ''
            lines.append(f'[{name}] {esc(t)}')
        lines.append('')
    out='\n'.join(lines)
    out_path=ROOT/'runs'/'sample'/'compare_samples.txt'
    out_path.write_text(out,encoding='utf-8')
    print(out)
    print(f'\nside-by-side saved to {out_path}')
if __name__=='__main__':
    main()
