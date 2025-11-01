# train_k_policy.py
import json, argparse, random, math, os, re
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ACTIONS = [2,3,4,5,6,8,10]

def toks(s):
    s = re.sub(r"[^a-z0-9]+"," ", (s or "").lower())
    return [w for w in s.split() if w]

def contains_gold(ctx: str, gold: str) -> bool:
    g = (gold or "").strip().strip("\"'")
    if len(g) <= 2: return False
    return g.lower() in (ctx or "").lower()

def reward_proxy(ctxs: List[str], gold: str, rer_scores: List[float], K: int) -> float:
    sel = ctxs[:K]
    hit = 1.0 if any(contains_gold(c, gold) for c in sel) else 0.0
    margin = 0.0
    if rer_scores:
        sub = rer_scores[:K]
        if len(sub) >= 2:
            top = max(sub); sec = max([x for x in sub if x != top], default=top)
            margin = max(0.0, top - sec)   # 0~1-ish
    return hit + 0.3 * margin

def basic_feats(query: str, ctxs: List[str]):
    q_len = len(toks(query)); num_ctx = len(ctxs)
    ctx_lens = [len(toks(c)) for c in ctxs]
    avg_ctx = sum(ctx_lens)/max(1, num_ctx) if num_ctx>0 else 0.0
    max_ctx = max(ctx_lens) if ctx_lens else 0.0
    def tokset(s): return set(toks(s))
    uniq_ratio = 0.0
    if num_ctx >= 2:
        u = len(tokset(" ".join(ctxs))); d = sum(len(tokset(c)) for c in ctxs)
        uniq_ratio = u / max(1, d)
    is_short = 1.0 if q_len <= 6 else 0.0
    f7 = sum(ctx_lens[:3])/max(1,len(ctx_lens[:3])) if ctx_lens else 0.0
    f8 = 1.0 if any("?" in (ctxs[i] if i<len(ctxs) else "") for i in range(min(2,len(ctxs)))) else 0.0
    return torch.tensor([q_len,num_ctx,avg_ctx,max_ctx,uniq_ratio,is_short,f7,f8], dtype=torch.float32)

class KPolicy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

def load_bandit(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            q = o.get("question") or ""
            gold = (o.get("answer") or {}).get("text","")
            ctxs = o.get("retrieved_texts") or []
            scores = o.get("reranker_scores") or []
            if q and ctxs:
                data.append({"q": q, "gold": gold, "ctxs": ctxs, "scores": scores})
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandit_file", type=str, default="./data/bandit_train.jsonl")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save_path", type=str, default="./models/k_policy.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_bandit(args.bandit_file)
    print(f"Loaded {len(data)} bandit items from {args.bandit_file}")

    policy = KPolicy(in_dim=8, out_dim=len(ACTIONS))
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    baseline = 0.0

    for ep in range(1, args.epochs+1):
        random.shuffle(data)
        tot_loss, tot_r = 0.0, 0.0
        for ex in tqdm(data, desc=f"EP{ep}"):
            q, gold, ctxs, scores = ex["q"], ex["gold"], ex["ctxs"], ex["scores"]
            x = basic_feats(q, ctxs).unsqueeze(0)  # [1,8]
            logits = policy(x)                     # [1, |A|]
            probs = torch.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs=probs)
            a_idx = m.sample().item()
            K = ACTIONS[a_idx]

            r = reward_proxy(ctxs, gold, scores, K)  # scalar
            baseline = 0.9 * baseline + 0.1 * r
            adv = r - baseline

            loss = -m.log_prob(torch.tensor(a_idx)) * adv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            tot_loss += float(loss.item())
            tot_r += float(r)

        print(f"[EP{ep}] avg_loss={tot_loss/len(data):.4f} avg_reward={tot_r/len(data):.4f} baseline={baseline:.3f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({"model": policy.state_dict(), "_in_dim": 8}, args.save_path)
    print(f"Saved to {args.save_path}")

if __name__ == "__main__":
    main()
