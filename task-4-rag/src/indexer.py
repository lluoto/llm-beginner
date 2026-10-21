from chunker import chunks_text
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass
import faiss
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def index_generator(sample,chunk_size,overlap):
    model=SentenceTransformer('model/bge-small-zh-v1.5',device=device)
    fragment_sen=chunks_text(sample,chunk_size,overlap)
    for pervious,current in zip(fragment_sen[:-1],fragment_sen[1:]):
        pervious_vector=model.encode(pervious,normalize_embeddings=True,batch_size=chunk_size,convert_to_tensor=True)
        current_vector=model.encode(current,normalize_embeddings=True,batch_size=chunk_size,convert_to_tensor=True)
        # cosine_value=torch.cosine_similarity(pervious,current,dim=-1)
        L2_pervious,L2_current=torch.nn.functional.normalize(pervious_vector,dim=-1),torch.nn.functional.normalize(current_vector,dim=-1)
        index=faiss.IndexFlatIP(L2_pervious.shape[-1])
        index.append(L2_pervious)
        dist,idx=index.search(L2_current)
        return idx