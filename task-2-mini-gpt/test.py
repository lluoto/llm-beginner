from src.tokenizer import BPETokenizer
BPE_processor=BPETokenizer.from_pretrained('ckpt/tokenizer.json',128)

ids = BPE_processor.encode("巴山上峡重复重")
text = BPE_processor.decode(ids)
print(text)