import torch 
from torch import nn
import math
def inject_lora(model,target_modules=None,r=8,alpha=16):
    target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "input_layernorm", "post_attention_layernorm", "embed_tokens", "lm_head", "norm"]
    for name,module in model.named_modules():
        target=name.split('.')[-1]
        if target in target_modules and isinstance(module,nn.Linear):
            module.weight.requires_grad=False
            if module.bias is not None:
                module.bias.requires_grad=False
            in_feat=module.in_features
            out_feat=module.out_features
            lora_a=torch.empty(r,in_feat,dtype=torch.bfloat16,device=module.weight.device)
            nn.init.kaiming_uniform_(lora_a,a=math.sqrt(5))
            lora_b=torch.zeros(out_feat,r,dtype=torch.bfloat16,device=module.weight.device)
            module.lora_A=nn.Parameter(lora_a)
            module.lora_B=nn.Parameter(lora_b)
            origin_forward=module.forward
            module.lora_scaling=alpha/r
            module._orig_forward=origin_forward
            def _lora_forward(self_,x,_orig=origin_forward):
                return _orig(x)+(alpha/r)*(x @ self_.lora_A.T @ self_.lora_B.T)
            module.forward=_lora_forward.__get__(module,type(module))
    for name,param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad=False
    return model
def merge_lora(model):
    for _,module in model.named_modules():
        if hasattr(module,'lora_A') and hasattr(module,'lora_B'):
            delta=module.lora_scaling*(module.lora_B@module.lora_A)
            module.weight.data+=delta.to(module.weight.device)
            del module.lora_A,module.lora_B
            module.forward=module._orig_forward
    return model
