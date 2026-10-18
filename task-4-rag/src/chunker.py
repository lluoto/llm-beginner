import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import torch
from collections import Counter
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
def chapter_spans(nodes,n_pages,max_level=0):
    tops=[n for n in nodes if n.level < max_level and n.page is not None]
    tops.sort(key=lambda n:n.page)
    spans=[]
    if tops and tops[0].page > 0:
        spans.append({"title": "__front_matter__", "start": 0, "end": tops[0].page})

    for i, n in enumerate(tops):
        start = n.page
        end = tops[i + 1].page if i + 1 < len(tops) else n_pages
        if end <= start:
            # Several headings share one page; keep the first, skip the rest
            continue
        spans.append({"title": n.title, "start": start, "end": end})
    return spans
def extract_chapters(pdf_path, max_level=0, layout=False):
    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        reader.decrypt("")               # try empty password first

    nodes = flatten_outline(reader)
    spans = chapter_spans(nodes, len(reader.pages), max_level)
    mode = "layout" if layout else "text"
    chapters = []
    for sp in spans:
        buf = []
        for p in range(sp["start"], sp["end"]):
            buf.append(reader.pages[p].extract_text(extraction_mode=mode) or "")
        text = "\n".join(buf)
        chapters.append({
            "title": sp["title"],
            "start_page": sp["start"] + 1,     # 1-based for humans
            "end_page": sp["end"],
            "chars": len(text),
            "text": text,
        })
    return chapters
def strip_running_lines(texts, min_ratio=0.5, max_len=60):
    """Remove lines that repeat on most pages (running headers/footers)."""
    cnt = Counter()
    for t in texts:
        for line in {l.strip() for l in t.splitlines() if l.strip()}:
            cnt[line] += 1
    n = len(texts)
    if n < 3:
        return texts
    bad = {l for l, c in cnt.items() if c / n >= min_ratio and len(l) <= max_len}
    return ["\n".join(l for l in t.splitlines() if l.strip() not in bad) for t in texts]
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    if overlap>chunk_size:
        raise ValueError('overlap must be smaller than size')
    model=SentenceTransformer('model/bge-small-zh-v1.5',device=device)
    
    if is_existing_file(sample):
        reader=PdfReader(sample)
        chapters=extract_chapters(sample,max_level=0,layout=False)
        cleaned=strip_running_lines([c['text'] for c in chapters])
        for c,t in zip(chapters,cleaned):
            c['text']=t
            c['chars']=len(t)
        for c in chapters:
            print(f"{c['title']} p{c['start_page']}-{c['end_page']} {c['chars']} chars")

        candidate_text=[chapter['text'] for chapter in chapters]
            
    else:
        candidate_text=sample

    if isinstance(candidate_text,type(list)):
        embed_vectors=[]

        for chap in candidate_text:
            chap_sentence=chap.strip('\n ').split('。')
            for i in range(len(chapters)):
                while len(candidate_text)<chunk_size:
                    overlap_start=chunk_size-overlap
                    current,candidate_text=candidate_text[:chunk_size],candidate_text[overlap_start:]
                    embed_vectors.append(model.encode(current,normalize_embeddings=True,batch_size=chunk_size,convert_to_tensor=True))
    else:
        candidate_text=candidate_text.strip('\n ').split('。')
        embed_vectors=[]
        for i in range(len(candidate_text)):
            while len(candidate_text)<chunk_size:
                overlap_start=chunk_size-overlap
                current,candidate_text=candidate_text[:chunk_size],candidate_text[overlap_start:]
                embed_vectors.append(model.encode(current,normalize_embeddings=True,batch_size=chunk_size,convert_to_tensor=True))
    return embed_vectors
