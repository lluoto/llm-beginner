import torch 
from torch import nn
import json
from pathlib import Path

class inject_lora(nn.Module):
    def __init__(self,model,target_modules=None,r=8,alpha=16,trainable_rate=0.03):
        super().__init__()
        self.model=model
        
        self.target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "input_layernorm", "post_attention_layernorm", "embed_tokens", "lm_head", "norm"]
        self.decode_layers=self.model.model.layers
        
        for name,module in self.model.named_modules():
            target=name.split('.')[-1]
            if target in self.target_modules and isinstance(module,nn.Linear):
                module.weight.requires_grad=False
                if module.bias is not None:
                    module.bias.requires_grad=False
                in_feat=module.in_features
                out_feat=module.out_features
                assert trainable_rate <0.05
                lora_a=torch.randn(self.r,in_feat,dtype=torch.bfloat16,device=module.weight.device)*trainable_rate
                lora_b=torch.zeros(out_feat,self.r,dtype=torch.bfloat16,device=module.weight.device)
                module.lora_A=nn.Parameter(lora_a)
                module.lora_B=nn.Parameter(lora_b)
                origin_forward=module.forward
                module.lora_scaleing=alpha/r
                module._orig_forward=origin_forward
                def _lora_forward(self_,x,_orig=origin_forward):
                    lora_out=(self.alpha/self.r)*(x @ self_.lora_B @ self_.lora_A)
                    return _orig(x)+lora_out
                module.forward=_lora_forward.__get__(module,type(module))
        for name,param in self.model.named_parameters():
            if 'lora' not in name:
                param.requires_grad=False

    def forward(self,*args,**kwargs):
        return self.model(*args,**kwargs)
def merge_lora(model):
    for _,module in model.named_modules():
        hasattr(module,'lora_A') and hasattr(module,'lora_B') 
        delta=(module.alpha/module.r)*(module.lora_A@module.lora_B)
        model.weight.data+=delta.to(module.weight.device)
        del module.lora_A,module.lora_B
        module.forward=module._orig_forward
    return model
