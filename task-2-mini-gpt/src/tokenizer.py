import json
import torch 
from torch import nn
from collections import Counter
import pandas as pd 
import ast

import re
code_format='utf-8'


    
class BPETokenizer(nn.Module):
    def __init__(self,embedding_dim,vocab_size,text_file,unk_token=1,pad_token=0,Dropout=0.1):
        super().__init__()
        self.unk_token=unk_token
        self.pad_token=pad_token
        self.drop=nn.Dropout(Dropout)
        self.embedding_dim=embedding_dim
        self.merged={}
        self.vocab,self.word2idx,self.idx2word=self.build_bpe_vocab(text_file,vocab_size)
        self.embedding=nn.Embedding(vocab_size,embedding_dim)

    def bpe_merge(self,all_text,process=100):
        stats=Counter()
        for sentence in all_text:
            token=[c.encode(code_format) for c in list(sentence)]
            stats.update(zip(token,token[1:]))
        candidate_list=[[c.encode(code_format) for c in list(s)]
                        for s in all_text]
        merged={}
        while process != 0:
            if not stats:
                break
            pair,_ =stats.most_common(1)[0]
            for si,stream in enumerate(candidate_list):
                new_stream=[]
                i=0
                while i < len(stream):
                    if (i< len(stream)-1
                        and stream[i]==pair[0]
                        and stream[i+1]==pair[1]):
                        new_stream.append(pair[0]+pair[1])
                        i+=2
                    else:
                        new_stream.append(stream[i])
                        i+=1

                candidate_list[si]=new_stream
            merged[pair]=pair[0]+pair[1]
            stats=Counter()
            for stream in candidate_list:
                stats.update(zip(stream,stream[1:]))
            process-=1
            
        self.merged=merged
        vocab=set()
        for stream in candidate_list:
            vocab.update(stream)
        vocab=sorted(set(vocab))
        return [i.decode(code_format) for i in vocab]
   
    def build_bpe_vocab(self,text_file,vocab_size=5000):
        text=[]
        with open(text_file,'r') as f:
            text=f.read()
        special_token=['<pad>','<unk>','<bos>','<eos>']

        text=[fragment for fragment in re.split(r'[，。\n]',text) if fragment!='']
        basal=[f'byte_{i}' for i in range(256)]
        n_merge=max(0,vocab_size-len(special_token)-len(basal))
        vocab=self.bpe_merge(text,process=n_merge)
        vocab=special_token+basal+vocab
        vocab=vocab[:vocab_size]

        word2idx={j:i for i,j in enumerate(vocab)}
        idx2word={i:j for i,j in enumerate(vocab)}
        self.vocab_size=len(vocab)
        return vocab,word2idx,idx2word
    def encode(self,text,max_len=None):
        candidnate_list=[c.encode(code_format) for c in list(text)]
        for pair,merge in self.merged.items():
            i=0
            new_token=[]
            while i < len(candidnate_list):
                if (i< len(candidnate_list)-1
                    and candidnate_list[i]==pair[0]
                    and candidnate_list[i+1]==pair[1]):
                    new_token.append(pair[0]+pair[1])
                    i+=2
                else:
                    new_token.append(candidnate_list[i])
                    i+=1

            candidnate_list=new_token
        
        ids=[]
        for token in candidnate_list:
            key=token.decode(code_format)
            if key in self.word2idx:

                ids.append(self.word2idx.get(key,self.unk_token))
            else:
                for b in token:
                    ids.append(self.word2idx.get(f"byte_{b}",self.unk_token))
        
        return ids
    
    def decode(self,ids):
        out=[]

        for i in ids:
            if isinstance(i,torch.Tensor):
                i=i.item()
            if i==self.pad_token:
                continue
            w=self.idx2word[i]
            if w.startswith('byte_'):
                out.append(bytes([int(w[5:])]))
            else:
                out.append(w.encode(code_format))

        return b''.join(out).decode(code_format,errors='replace')
    def save_pretrained(self,dir='ckpt'):
        with open(f'{dir}/tokenizer.json','w') as f:
            json.dump(self.word2idx,f,ensure_ascii=False,indent=2)
        merge_json={}
        for (a,b),m in self.merged.items(): 
            if isinstance(a,bytes) and isinstance(b,bytes):
                k=[list(a),list(b)]
                merge_json[str(k)]=list(m)
        with open(f'{dir}/merged.json','w',encoding=code_format) as f:
            json.dump(merge_json,f,ensure_ascii=False,indent=2)
    def forward(self,text,max_len=None):
        ids=self.encode(text,max_len)
        return self.embedding(torch.tensor(ids,dtype=torch.long))
        
    @classmethod
    def from_pretrained(cls,dir,embedding_dim=128):
        dim= embedding_dim if embedding_dim is not None else 128
        self=cls.__new__(cls)
        nn.Module.__init__(self)
        with open(dir,'r') as f:
            word2idx=json.load(f)
        self.word2idx=word2idx
        self.idx2word={int(self.word2idx[i]):i for i in word2idx.keys()}
        self.vocab=list(word2idx.keys())
        with open(dir.replace('tokenizer.json','merged.json'),'r') as f:
            merged_json=json.load(f)
        self.merged={}
        for key,meg_str in merged_json.items():
            a_str,b_str=eval(key)
            a=bytes(a_str)
            b=bytes(b_str)
            self.merged[(a,b)]=bytes(meg_str)
        self.embedding_dim = dim
        self.embedding = nn.Embedding(len(word2idx), dim)   # ← 重建 embedding
        self.drop = nn.Dropout(0.1)
        self.unk_token = 1
        self.pad_token = 0
        if embedding_dim is not None and self.embedding.embedding_dim==embedding_dim:
            pass
        else:
            self.embedding=nn.Embedding(len(self.word2idx),self.embedding.embedding_dim) 
        return self

