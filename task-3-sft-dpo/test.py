import torch 
from torch import nn
import json
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT = Path(__file__).resolve().parents[0]
model_path = ROOT / "models" / "Qwen2.5-0.5B"
tokenizer= AutoTokenizer.from_pretrained(str(model_path))
help(tokenizer)
# model = AutoModelForCausalLM.from_pretrained(
#     "models/Qwen2.5-0.5B",
#     torch_dtype=torch.float32,   # 或 "auto"
#     device_map="auto",           # 或指定 device
#     low_cpu_mem_usage=True,
# )
# layer=model.model.layers[0]
# def get_qkv_edge(layer,target_model):
#     return [param.shape[0] for name,param in layer.self_attn.named_parameters() if name==f'{target_model}.bias'][0]
# for name, param in layer.self_attn.named_parameters():
#     print(f"  {name:40s} | shape={list(param.shape)}")

# print("\n第5层 mlp 参数：")
# for name, param in layer.mlp.named_parameters():
#     print(f"  {name:40s} | shape={list(param.shape)}")