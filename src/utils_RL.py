
from typing import List
import re
#RL version
# --- utils.py 內新增：K policy 接入 ---
import os
import math
import torch
import torch.nn as nn
from typing import List

_FINAL_PAT = re.compile(r"final\s*[_\s-]*answer\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)

def get_inference_system_prompt() -> str:
    return (
        "You are a careful QA assistant. Answer ONLY using the provided context. "
        "If the answer is not present, output 'CANNOTANSWER'. Be concise."
    )

def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    # ★ RL 選 K（若無 policy，退回 K=5）
    K = _choose_k(query, context_list)
    context_list = context_list[:K]

    ctx = "\n\n".join([f"[CTX {i+1}]\n{c}" for i, c in enumerate(context_list)])
    return (
        f"Question: {query}\n\n"
        f"Context passages:\n{ctx}\n\n"
        "Instructions:\n"
        "1) Use only the context to answer.\n"
        "2) If the answer is missing, output exactly: CANNOTANSWER.\n"
        "3) Reply in the format:\n"
        "final_answer: <your answer here>"
    )



# def parse_generated_answer(pred_ans: str) -> str:
#     """Extract the actual answer from the model's generated text."""
#     parsed_ans = pred_ans
#     return parsed_ans

# def parse_generated_answer(pred_ans: str) -> str:
#     m = re.search(r"final_answer:\s*(.+)", pred_ans, flags=re.IGNORECASE|re.DOTALL)
#     ans = m.group(1).strip() if m else pred_ans.strip()
#     ans = re.split(r"\n(?:Thought|Context|Question):", ans)[0].strip()
#     return ans

def parse_generated_answer(pred_ans: str) -> str:
    text = (pred_ans or "").strip()
    match = None
    for m in _FINAL_PAT.finditer(text):
        match = m
    ans = match.group(1).strip() if match else ""
    if len(ans) >= 2 and (ans[0] == ans[-1]) and ans[0] in "\"'“”‘’":
        ans = ans[1:-1].strip()
    return ans if ans else "CANNOTANSWER"



_K_ACTIONS = [2, 3, 4, 5, 6, 8, 10]  
_K_DEFAULT = 5
_K_POLICY_PATH = "./models/k_policy.pt"

class KPolicy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.net(x)  # logits

# 載入（或回退為 None）
_kpolicy = None
def _load_kpolicy():
    global _kpolicy
    if _kpolicy is not None:
        return _kpolicy
    if os.path.exists(_K_POLICY_PATH):
        state = torch.load(_K_POLICY_PATH, map_location="cpu")
        in_dim = state.get("_in_dim", 8)
        out_dim = len(_K_ACTIONS)
        model = KPolicy(in_dim, out_dim)
        model.load_state_dict(state["model"])
        model.eval()
        _kpolicy = (model, in_dim)
    else:
        _kpolicy = None
    return _kpolicy

def _basic_feats(query: str, ctxs: List[str]) -> torch.Tensor:
    """超輕量特徵（只用文字，不需 retriever 分數）"""
    q_len = len(query.split())
    num_ctx = len(ctxs)
    ctx_lens = [len(c.split()) for c in ctxs]
    avg_ctx = sum(ctx_lens)/max(1, num_ctx)
    max_ctx = max(ctx_lens) if ctx_lens else 0
    # 粗略多樣性：去重 token 占比
    def tokset(s):
        return set([t for t in "".join(ch if ch.isalnum() else " " for ch in s.lower()).split()])
    uniq_ratio = 0.0
    if num_ctx >= 2:
        u = len(tokset(" ".join(ctxs)))
        d = sum(len(tokset(c)) for c in ctxs)
        uniq_ratio = u / max(1, d)
    # 問題是否很短（短問常需較少 K）
    is_short = 1.0 if q_len <= 6 else 0.0
    feats = torch.tensor([
        q_len, num_ctx, avg_ctx, max_ctx, uniq_ratio, is_short,
        sum(ctx_lens[:3])/max(1,len(ctx_lens[:3])),  # 前3段平均長度
        1.0 if any("?" in c for c in ctxs[:2]) else 0.0  # 少量噪音特徵
    ], dtype=torch.float32)
    return feats

def _choose_k(query: str, ctxs: List[str]) -> int:
    loaded = _load_kpolicy()
    if not loaded:
        return _K_DEFAULT
    model, in_dim = loaded
    feats = _basic_feats(query, ctxs)
    if feats.shape[0] != in_dim:  # 兼容：若特徵維度不符，回退
        return _K_DEFAULT
    with torch.no_grad():
        logits = model(feats.unsqueeze(0))
        k_idx = int(torch.argmax(logits, dim=-1).item())
    return _K_ACTIONS[min(max(k_idx, 0), len(_K_ACTIONS)-1)]
