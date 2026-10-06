from torch import nn
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

start_id=151644
end_id=151645
user=77091
assistant=872
def format_messages(msgs):
    round_chat=['<|im_start|>system\nYou are a helpful assistant<|im_end|>']
    for message in msgs:
        round_chat.append(
'<|im_start|>{role}\n{content}<|im_end|>\n'.format(role=message['role'],content=message['content'])
            )

    round_chat.append('')
    return "\n".join(round_chat)
def build_labels(ids,msgs,max_len=64):

    if not msgs or max_len<1:
        raise ValueError('对话为空，max需要为正')
    labels=ids.clone()
    labels[:]=-100
    in_assistant=False
    megs_id=-1
    for i in range(len(ids)):
        if ids[i]==start_id:
            if i+1 < len(ids) and ids[i+1]==tok(msgs[megs_id]['role'],return_tensors="pt").input_ids[0]:
                in_assistant=True
            else:
                in_assistant=False
            labels[i]=-100
            megs_id+=1
        elif ids[i]==end_id:
            in_assistant=False
            labels[i]=-100
        elif in_assistant:
            labels[i]=ids[i]
    return labels