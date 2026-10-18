import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass
from collections import Counter
def chapter_spans(nodes,n_pages,max_level=0):
    tops=[n for n in nodes if n.level < max_level and n.page is not None]
    tops.sort(key=lambda n:n.page)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def flatten_outline(reader,outline=None,level=0):
    if outline is None:
        outline=reader.outline
    nodes=[]
    for item in outline:
        if isinstance(item,list):
            nodes.extend(flatten_outline(reader,item,level+1))
        else:
            title=(getattr(item,'title','') or '').strip()
            try:
                page=reader.get_destination_page_number(item)
            except Exception:
                page=None
            nodes.append(None(level,title,page))
    return nodes
def is_existing_file(obj):
    if obj is None:
        return False
    try:
        path=os.fspath(obj)
    except TypeError:
        path=None
    if path is not None:
        if not path:
            return False
        return os.path.isfile(path)
    name=getattr(obj,'name',None)
    if isinstance(name,(str,bytes,os.PathLike)):
        return bool(name) and os.path.isfile(name)
    return False
def chunks_text(sample,chunk_size,overlap=32):
    model=SentenceTransformer('model/bge-small-zh-v1.5',device=device)
    
    if is_existing_file(sample):
        reader=PdfReader(sample)
        candidate_text=''
        for page in reader.pages:
            candidate_text+=page.extract_text() or ""
    else:
        candidate_text=sample
    candidate_text=candidate_text.strip('\n ').split('。')
    for i in range(1,len(candidate_text)):
    # while len(candidate_text)<chunk_size:
    #     current,candidate_text=candidate_text[:chunk_size],candidate_text[chunk_size+1:]
    #     current_embed=model.encode(current,normalize_embeddings=True,batch_size=chunk_size,convert_to_tensor=True)

