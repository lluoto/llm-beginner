import argparse
import torch
import random
from src.tokenizer import BPETokenizer
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.model import load_for_eval
from pathlib import Path
import os 
from time import time
from src.tokenizer import BPETokenizer
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
start_time=time()
dataset='simple'
# dataset='skypile'
runtime_writer=SummaryWriter(f'runs/{dataset}/sample/kv_cache')


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class read_dataset():
    def __init__(self,file_dir,model,max_len=512,pad_token_id=0):
        super().__init__()
        text=open(file_dir,'r',encoding='utf-8').read()
        self.text=text.split('\n\n')
        self.model=model
        self.max_len=max_len
        self.pad_token=pad_token_id
        self.block=[]
        for paragraph in self.text:
            input_tensor=torch.tensor(self.model.encode(paragraph,self.max_len),dtype=torch.long)

            if len(input_tensor)<=self.max_len:
                padded=torch.cat([
                    input_tensor,
                    torch.full((self.max_len-len(input_tensor),),self.pad_token,dtype=torch.long)
                ])
                self.block.append(padded)
            else:
                stride=self.max_len
                if dataset=='simple':
                    stride=self.max_len//8

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
        parser.add_argument('--test_dir','-test',type=str,default='data/test.txt')
    else:
        parser.add_argument('--test_dir','-test',type=str,default='data/simple/poetry.txt')
    parser.add_argument('--batch_size','-b',type=int,default='16')
    parser.add_argument('--max_len','-max',type=int,default=64)
    return parser
def hyper_para_output(model,index,input,return_cache):
    if index==0:
        return model.generate(input,temperature=0,top_k=None,top_p=None,mask=None,return_cache=return_cache)
    elif index==1:
        return model.generate(input,temperature=0.8,top_k=None,top_p=None,mask=None,return_cache=return_cache)
    elif index==2:
        return model.generate(input,temperature=0.2,top_k=None,top_p=None,mask=None,return_cache=return_cache)
    elif index==3:
        return model.generate(input,temperature=0,top_k=30,top_p=None,mask=None,return_cache=return_cache)
    elif index==4:
        return model.generate(input,temperature=0,top_k=None,top_p=0.9,mask=None,return_cache=return_cache)
    elif index==5:
        return model.generate(input,temperature=0.8,top_k=None,top_p=0.9,mask=None,return_cache=return_cache)
    elif index==6:
        return model.generate(input,temperature=0.8,top_k=30,top_p=None,mask=None,return_cache=return_cache)
    elif index==7:
        return model.generate(input,temperature=0.2,top_k=30,top_p=None,mask=None,return_cache=return_cache)


def main():
    args=parser_para()
    args=parser_para().parse_args()
    ckpt=  "ckpt/best.pt"
    model,BPE_processor=load_for_eval(str(ckpt))
    test_set=DataLoader(dataset=read_dataset(args.test_dir,BPE_processor,args.max_len),batch_size=args.batch_size)

    model.to(device)
    model.eval()
    return_cache=False
    for switch in range(2):
        if switch==1:
            return_cache=True
        for i in range(8):
            with torch.no_grad():
                for batch in test_set:
                    input,target=batch[:,:-1],batch[:,1:]
                    input,target=input.to(device),target.to(device)
                    output=hyper_para_output(model,i,input,return_cache)
                    poem=BPE_processor.decode(output[0])
                    print(poem,f'index group:{i}')
                    # break
        t = time() - start_time
        print("time consumption =", t, type(t))
        runtime_writer.add_hparams({'mode':f'kv_cache:{return_cache}'},{'timeconsumption':t})
if __name__=='__main__':
    main()