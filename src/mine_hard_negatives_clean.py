# mine_hard_negatives_clean.py
import json, faiss, numpy as np, re
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from data_utils import load_corpus, load_qrels, load_train

TOPK = 50
NEG_PER_POS = 7  # 1 pos + 7 neg → group size 8

def normalize_text(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return [w for w in s.split() if w]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def main():
    corpus = load_corpus("./data/corpus.txt")  # pid -> {text,title}
    qrels = load_qrels("./data/qrels.txt")
    train_items = load_train("./data/train.txt")

    pids = list(corpus.keys())
    texts = []
    for pid in pids:
        it = corpus[pid]
        ptxt = it.text
        if it.title:
            ptxt = f"{it.title}\n{ptxt}"
        texts.append(f"passage: {ptxt}")

    retriever = SentenceTransformer("intfloat/multilingual-e5-small", trust_remote_code=True)
    p_emb = retriever.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

    faiss.normalize_L2(p_emb)
    index = faiss.IndexFlatIP(p_emb.shape[1])
    index.add(p_emb.astype(np.float32))

    out = Path("./data/reranker_mined_clean.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for ex in tqdm(train_items, desc="Mining"):
            qid = ex.get("qid")
            q = ex.get("rewrite") or ex.get("question") or ""
            if not q or qid not in qrels: 
                continue

            # 取一個正樣本文本
            pos_text = None
            for pid in qrels[qid]:
                if pid in corpus:
                    it = corpus[pid]
                    pos_text = f"{it.title}\n{it.text}" if it.title else it.text
                    break
            if pos_text is None:
                # 後備：從 evidences label==1
                evs = ex.get("evidences") or []
                labs = ex.get("retrieval_labels") or []
                for e, l in zip(evs, labs):
                    if l == 1:
                        pos_text = e
                        break
            if pos_text is None:
                continue

            q_emb = retriever.encode([f"query: {q}"], convert_to_numpy=True, normalize_embeddings=True)
            D, I = index.search(q_emb.astype(np.float32), TOPK)

            gold = set(qrels[qid])
            pos_tok = normalize_text(pos_text)
            negs = []
            for idx in I[0]:
                pid = pids[idx]
                if pid in gold: 
                    continue
                cand = corpus[pid]
                cand_txt = f"{cand.title}\n{cand.text}" if cand.title else cand.text
                # 過濾高度重疊的假負樣本
                if jaccard(pos_tok, normalize_text(cand_txt)) >= 0.6:
                    continue
                negs.append(cand_txt)
                if len(negs) >= NEG_PER_POS:
                    break
            if len(negs) < NEG_PER_POS:
                continue

            rec = {"query": q, "pos": pos_text, "negs": negs}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
