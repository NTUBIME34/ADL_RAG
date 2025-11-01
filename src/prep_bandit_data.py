# prep_bandit_data.py
import json, argparse, os
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def read_corpus(corpus_file: str) -> List[str]:
    texts = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # 以 text 為主；若你想帶 title 可自行拼接
            t = obj.get("text") or ""
            texts.append(t)
    return texts

def encode_corpus(model: SentenceTransformer, corpus_texts: List[str], batch_size=256) -> np.ndarray:
    embs = model.encode(corpus_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    return embs.astype("float32")

def build_faiss_index(embs: np.ndarray):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    return index

def bi_encode_queries(model: SentenceTransformer, queries: List[str], batch_size=64) -> np.ndarray:
    q = model.encode(queries, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    return q.astype("float32")

def topk_search(index, qvecs, k):
    D, I = index.search(qvecs, k)
    return D, I  # cosine (IP on normalized vectors)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, required=True)   # e.g., ./data/train.txt
    ap.add_argument("--corpus_file", type=str, required=True)  # e.g., ./data/corpus.txt
    ap.add_argument("--retriever_ckpt", type=str, required=True) # e.g., intfloat/multilingual-e5-small or your fine-tuned path
    ap.add_argument("--reranker_ckpt", type=str, default="")     # 可空；e.g., cross-encoder/ms-marco-MiniLM-L-12-v2
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out_file", type=str, default="./data/bandit_train.jsonl")
    args = ap.parse_args()

    print("Loading corpus ...")
    corpus_texts = read_corpus(args.corpus_file)
    print(f"corpus size: {len(corpus_texts)}")

    print("Loading retriever bi-encoder ...")
    biencoder = SentenceTransformer(args.retriever_ckpt)

    print("Encoding corpus ...")
    c_embs = encode_corpus(biencoder, corpus_texts)
    index = build_faiss_index(c_embs)

    # optional reranker
    reranker = None
    if args.reranker_ckpt:
        print("Loading reranker cross-encoder ...")
        reranker = CrossEncoder(args.reranker_ckpt, max_length=512)

    # 遍歷 train queries
    fw = open(args.out_file, "w", encoding="utf-8")

    batch_q = []
    batch_meta = []
    for obj in load_jsonl(args.train_file):
        q = obj.get("question") or obj.get("rewrite") or ""
        gold = (obj.get("answer") or {}).get("text", "")
        if not q: continue
        batch_q.append(q)
        batch_meta.append({"gold": gold})

        # 批次處理
        if len(batch_q) >= 128:
            q_embs = bi_encode_queries(biencoder, batch_q)
            _, I = topk_search(index, q_embs, args.topk)
            for qi, idxs in enumerate(I):
                texts = [corpus_texts[j] for j in idxs]
                scores = []
                if reranker:
                    pairs = [(batch_q[qi], t) for t in texts]
                    s = reranker.predict(pairs, convert_to_numpy=True).tolist()
                    scores = s
                out = {
                    "question": batch_q[qi],
                    "answer": {"text": batch_meta[qi]["gold"]},
                    "retrieved_texts": texts,
                    "reranker_scores": scores
                }
                fw.write(json.dumps(out, ensure_ascii=False) + "\n")
            batch_q, batch_meta = [], []

    # 殘批
    if batch_q:
        q_embs = bi_encode_queries(biencoder, batch_q)
        _, I = topk_search(index, q_embs, args.topk)
        for qi, idxs in enumerate(I):
            texts = [corpus_texts[j] for j in idxs]
            scores = []
            if reranker:
                pairs = [(batch_q[qi], t) for t in texts]
                s = reranker.predict(pairs, convert_to_numpy=True).tolist()
                scores = s
            out = {
                "question": batch_q[qi],
                "answer": {"text": batch_meta[qi]["gold"]},
                "retrieved_texts": texts,
                "reranker_scores": scores
            }
            fw.write(json.dumps(out, ensure_ascii=False) + "\n")

    fw.close()
    print(f"Saved bandit file to {args.out_file}")

if __name__ == "__main__":
    main()
