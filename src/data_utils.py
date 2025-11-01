# data_utils.py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
from dataclasses import dataclass

@dataclass
class CorpusItem:
    pid: str
    text: str
    title: Optional[str] = None

def _read_json_or_jsonl(path: Path):
    """
    支援:
    - JSON 檔 (單一 dict or list)
    - JSONL 檔 (每行一個 JSON 物件)
    """
    text = path.read_text(encoding="utf-8").strip()
    # 嘗試當成單一 JSON
    try:
        obj = json.loads(text)
        # 若是字典或清單，直接回傳包裝成 list
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    # 嘗試逐行 JSONL
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def load_corpus(corpus_path: str) -> Dict[str, CorpusItem]:
    """
    corpus.txt 每行: {"text": "...", "title": "...", "id": "26054972@0", ...}
    回傳 pid -> CorpusItem
    """
    corpus = {}
    for obj in _read_json_or_jsonl(Path(corpus_path)):
        pid = obj.get("id") or obj.get("pid") or obj.get("docid")
        txt = obj.get("text") or obj.get("contents") or obj.get("body") or ""
        title = obj.get("title")
        if not pid or not txt:
            continue
        corpus[pid] = CorpusItem(pid=pid, text=txt, title=title)
    if not corpus:
        raise ValueError(f"No passages loaded from {corpus_path}")
    return corpus

def load_qrels(qrels_path: str) -> Dict[str, List[str]]:
    """
    qrels.txt 可能是:
    1) JSON (dict): { "qid": {"pid": 1, ...}, ... }
    2) JSONL 多行，每行一個 dict
    3) 或者多個 dict 合在一個 list
    回傳: qid -> [positive_pid, ...]
    """
    # 合併所有物件為一個 dict
    items = _read_json_or_jsonl(Path(qrels_path))
    merged = {}
    for obj in items:
        if isinstance(obj, dict):
            for k, v in obj.items():
                merged[k] = v
    qrels = {}
    for qid, pid_map in merged.items():
        pos = []
        if isinstance(pid_map, dict):
            for pid, rel in pid_map.items():
                if rel == 1 or rel == "1" or rel is True:
                    pos.append(pid)
        elif isinstance(pid_map, list):
            # 如果是 list of (pid, rel) 或 list of pid
            for it in pid_map:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    if it[1] == 1:
                        pos.append(it[0])
                elif isinstance(it, str):
                    pos.append(it)
        qrels[qid] = pos
    return qrels

def load_train(train_path: str) -> List[dict]:
    """
    train.txt 每行一個 JSON:
    {
      "qid": "...",
      "rewrite": "...",
      "evidences": [str, str, ...],
      "retrieval_labels": [0/1, ...],
      ...
    }
    """
    return _read_json_or_jsonl(Path(train_path))

def build_retriever_pairs(
    train_items: List[dict],
    corpus: Dict[str, CorpusItem],
    qrels: Dict[str, List[str]],
    prefer_qrels: bool = True
) -> List[Tuple[str, str]]:
    """
    產生 retriever 的 (query, positive_passage_text) pair。
    - 優先用 qrels 裡正樣本 pid 找 corpus 文字
    - 若找不到，退回用 train.evidences 中 label==1 的那段文字做 positive
    MultipleNegativesRankingLoss 會用同 batch 彼此當負樣本
    """
    pairs = []
    miss_by_qrels = 0
    for ex in train_items:
        qid = ex.get("qid")
        query = ex.get("rewrite") or ex.get("question") or ""
        if not query:
            continue
        pos_text = None
        if prefer_qrels and qid in qrels and len(qrels[qid]) > 0:
            # 取第一個正 pid
            for pid in qrels[qid]:
                if pid in corpus:
                    pos_text = corpus[pid].text
                    break
        if pos_text is None:
            # 從 evidences 裡找 label==1
            evs = ex.get("evidences") or []
            labs = ex.get("retrieval_labels") or []
            for e, l in zip(evs, labs):
                if l == 1:
                    pos_text = e
                    break
            if pos_text is None:
                miss_by_qrels += 1
                continue
        pairs.append((query, pos_text))
    if miss_by_qrels > 0:
        print(f"[build_retriever_pairs] Fallback positives used:", miss_by_qrels)
    return pairs

def build_reranker_samples(
    train_items: List[dict],
) -> List[Tuple[str, str, float]]:
    """
    產生 reranker 的 (query, passage_text, label) 樣本。
    直接用 evidences / retrieval_labels。
    """
    samples = []
    for ex in train_items:
        query = ex.get("rewrite") or ex.get("question") or ""
        if not query:
            continue
        evs = ex.get("evidences") or []
        labs = ex.get("retrieval_labels") or []
        for e, l in zip(evs, labs):
            lbl = 1.0 if (l == 1 or l is True) else 0.0
            samples.append((query, e, lbl))
    return samples
