from transformers import AutoTokenizer
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
model_path = ROOT / "models" / "Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(str(model_path))

start_id=151644
end_id=151645

def format_messages(msgs):
    round_chat=['<|im_start|>system\nYou are a helpful assistant<|im_end|>\n']
    for message in msgs:
        round_chat.append(
'<|im_start|>{role}\n{content}<|im_end|>\n'.format(role=message['role'],content=message['content'])
            )

    return "".join(round_chat)
def build_labels(ids,msgs,max_len=64):

    if not msgs or max_len<1:
        raise ValueError('对话为空，max需要为正')
    labels=ids.clone()
    labels[:]=-100
    in_assistant=False
    megs_id=-1

    for i in range(len(ids)):
        if ids[i]==start_id:
            role=msgs[megs_id]['role'] if 0 < megs_id<len(msgs) else None
            in_assistant=(role=='assistant')
            labels[i]=-100
            megs_id+=1
        elif ids[i]==end_id:
            in_assistant=False
            labels[i]=-100
        elif in_assistant:
            labels[i]=ids[i]
    return labels