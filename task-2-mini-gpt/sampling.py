import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.model import load_for_eval
import os
from time import perf_counter
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

dataset='simple'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAMPLING_CONFIGS = [
    {'name': 'greedy',                 'temperature': 0.0, 'top_k': None, 'top_p': None},
    {'name': 'T0.2',                   'temperature': 0.2, 'top_k': None, 'top_p': None},
    {'name': 'T0.8',                   'temperature': 0.8, 'top_k': None, 'top_p': None},
    {'name': 'T0.2+k30',               'temperature': 0.2, 'top_k': 30,   'top_p': None},
    {'name': 'T0.2+p0.9',              'temperature': 0.2, 'top_k': None, 'top_p': 0.9},
    {'name': 'T0.8+p0.9',              'temperature': 0.8, 'top_k': None, 'top_p': 0.9},
    {'name': 'T0.8+k30',               'temperature': 0.8, 'top_k': 30,   'top_p': None},
    {'name': 'T0.2+k30+p0.9',          'temperature': 0.2, 'top_k': 30,   'top_p': 0.9},
    {'name': 'T1.0+k50+p0.95',         'temperature': 1.0, 'top_k': 50,   'top_p': 0.95},
]

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
    parser=argparse.ArgumentParser(description='mini-GPT sampling sweep + KV cache benchmark')
    if dataset=='skypile':
        parser.add_argument('--test_dir','-test',type=str,default='data/test.txt')
    else:
        parser.add_argument('--test_dir','-test',type=str,default='data/simple/poetry.txt')
    parser.add_argument('--batch_size','-b',type=int,default=16)
    parser.add_argument('--max_len','-max',type=int,default=64)
    parser.add_argument('--n_samples','-n',type=int,default=1)

    parser.add_argument('--temperature','-t',type=float,default=None)
    parser.add_argument('--top_k','-k',type=int,default=None)
    parser.add_argument('--top_p','-p',type=float,default=None)
    parser.add_argument('--prompt',type=str,default=None)

    parser.add_argument('--bench',action='store_true')
    parser.add_argument('--bench_prompt',type=str,default='床前明月光，疑是地上霜。举头望明月，低头思故乡。')
    parser.add_argument('--n_new',type=int,nargs='+',default=[64,128,256])
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--warmup',type=int,default=2)
    return parser


def build_configs(args):
    if args.temperature is not None or args.top_k is not None or args.top_p is not None:
        return [{'name':f'T{args.temperature}+k{args.top_k}+p{args.top_p}',
                 'temperature':args.temperature if args.temperature is not None else 1.0,
                 'top_k':args.top_k,'top_p':args.top_p}]
    return SAMPLING_CONFIGS


def benchmark_kv_cache(model,prompt,n_new=128,repeats=5,warmup=2):
    def _run(use_cache):
        with torch.no_grad():
            for _ in range(warmup):
                model.generate(prompt,temperature=0,return_cache=use_cache,max_len=n_new)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0=perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                model.generate(prompt,temperature=0,return_cache=use_cache,max_len=n_new)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (perf_counter()-t0)/repeats
    off=_run(False)
    on=_run(True)
    return off,on


def run_benchmark(model,BPE_processor,args,writer):
    prompt=torch.tensor([BPE_processor.encode(args.bench_prompt,512)],dtype=torch.long).to(device)
    for n_new in args.n_new:
        off,on=benchmark_kv_cache(model,prompt,n_new=n_new,repeats=args.repeats,warmup=args.warmup)
        print(f'[S3] max_new={n_new:4d}  OFF={off*1e3:8.2f} ms  ON={on*1e3:8.2f} ms  speedup={off/on:5.2f}x',flush=True)
        writer.add_hparams({'mode':'kv_cache_bench','n_new':n_new},
                           {'off_s':off,'on_s':on,'speedup':off/on},
                           run_name=f'bench_{n_new}')
        writer.flush()


def run_sweep(model,BPE_processor,test_set,configs,max_len,n_samples):
    for return_cache in (False,True):
        for config in configs:
            for bi,batch in enumerate(test_set):
                if n_samples and bi>=n_samples:
                    break
                inp=batch[:,:-1].to(device)
                with torch.no_grad():
                    output=model.generate(inp,
                                          temperature=config['temperature'],
                                          top_k=config['top_k'],
                                          top_p=config['top_p'],
                                          return_cache=return_cache,
                                          max_len=max_len)
                print(BPE_processor.decode(output[0]),f"[[{config['name']} cache={return_cache}]]",flush=True)


def main():
    args=parser_para().parse_args()
    model,BPE_processor=load_for_eval('ckpt/best.pt')
    model.to(device)
    model.eval()
    writer=SummaryWriter(f'runs/{dataset}/sample/kv_cache')

    if args.bench:
        run_benchmark(model,BPE_processor,args,writer)

    configs=build_configs(args)
    if args.prompt is not None:
        prompt_ids=torch.tensor([BPE_processor.encode(args.prompt,512)],dtype=torch.long).to(device)
        for return_cache in (False,True):
            for config in configs:
                with torch.no_grad():
                    output=model.generate(prompt_ids,
                                          temperature=config['temperature'],
                                          top_k=config['top_k'],
                                          top_p=config['top_p'],
                                          return_cache=return_cache,
                                          max_len=args.max_len)
                print(BPE_processor.decode(output[0]),f"[[{config['name']} cache={return_cache}]]",flush=True)
    else:
        test_set=DataLoader(dataset=read_dataset(args.test_dir,BPE_processor,args.max_len),batch_size=args.batch_size)
        run_sweep(model,BPE_processor,test_set,configs,args.max_len,args.n_samples)

    writer.flush()
    writer.close()


if __name__=='__main__':
    main()
