# train_reranker_listwise.py — curriculum + multi-positive + listwise CE + pairwise margin
import os, json, time, argparse, math, random, sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# --------------------------
# Utilities
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_curriculum(spec: str, default_group_size: int, total_epochs: int) -> List[int]:
    """
    解析字串，例如: "1x4,3x8" → [4, 8, 8, 8]
    若未提供或不足，就用 default_group_size 補到 total_epochs。
    """
    if not spec:
        return [default_group_size] * total_epochs
    plan = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "x" in part:
            n, g = part.split("x")
            n = int(n.strip()); g = int(g.strip())
            plan.extend([g] * n)
        else:
            plan.append(int(part))
    if len(plan) < total_epochs:
        plan.extend([default_group_size] * (total_epochs - len(plan)))
    return plan[:total_epochs]

# --------------------------
# Datasets
# --------------------------
class BaseMined:
    """
    持有完整 mined 資料。
    每行支援：
      {"query": str, "pos": str, "negs": [str, ...]}
    或（可選）：
      {"query": str, "pos_list": [str, ...], "negs": [str, ...]}
    """
    def __init__(self, mined_path: str, shuffle_pos: bool = True):
        self.items = []
        with open(mined_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                q = obj["query"]
                pos_list = obj.get("pos_list")
                if pos_list is None:
                    pos_list = [obj["pos"]]
                negs = obj["negs"]
                self.items.append({"q": q, "pos_list": pos_list, "negs": negs})
        self.shuffle_pos = shuffle_pos

    def __len__(self):
        return len(self.items)

    def sample_group(
        self,
        idx: int,
        group_size: int,
        max_pos_in_group: int = 1,
    ) -> Dict:
        """
        從第 idx 筆資料動態組一個 group：
          - 取最多 max_pos_in_group 個正樣本（若不足就用 1）
          - 再隨機補 (group_size - #pos) 個負樣本
        回傳：
          {"q": str, "cands": [str,...], "pos_indices": [int, ...]}
        """
        it = self.items[idx]
        q = it["q"]
        pos_list = it["pos_list"][:]
        negs = it["negs"]

        # 選正樣本（可隨機打散）
        if self.shuffle_pos:
            random.shuffle(pos_list)
        pos_list = pos_list[: max(1, min(max_pos_in_group, len(pos_list)))]

        need_negs = max(1, group_size - len(pos_list))
        if need_negs > len(negs):
            # 若負樣本不夠，重複抽；一般不會發生
            chosen_negs = (negs * (math.ceil(need_negs / max(1, len(negs)))))[:need_negs]
        else:
            chosen_negs = random.sample(negs, need_negs)

        cands = pos_list + chosen_negs
        pos_indices = list(range(0, len(pos_list)))  # 正樣本在前面
        return {"q": q, "cands": cands, "pos_indices": pos_indices}

class EpochDataset(Dataset):
    """
    每個 epoch 建立一次，從 BaseMined 動態重採樣（可改變 group_size / max_pos_in_group）。
    """
    def __init__(self, base: BaseMined, group_size: int, max_pos_in_group: int):
        self.base = base
        self.group_size = group_size
        self.max_pos_in_group = max_pos_in_group

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base.sample_group(idx, self.group_size, self.max_pos_in_group)

def make_collate(tokenizer, max_len: int):
    def collate_fn(batch: List[Dict]):
        # 扁平化 tokenization，記錄每組切片與正樣本索引
        all_q, all_c = [], []
        slices: List[Tuple[int, int]] = []
        pos_in_group: List[List[int]] = []
        offset = 0
        for item in batch:
            q = item["q"]
            cs = item["cands"]
            pis = item["pos_indices"]
            for c in cs:
                all_q.append(q)
                all_c.append(c)
            slices.append((offset, offset + len(cs)))
            pos_in_group.append(pis)
            offset += len(cs)

        enc = tokenizer(
            all_q, all_c,
            padding=True, truncation=True, max_length=max_len,
            return_tensors="pt"
        )
        return enc, slices, pos_in_group
    return collate_fn

# --------------------------
# Losses
# --------------------------
def listwise_soft_ce_loss(
    logits: torch.Tensor,            # [L] for one group
    pos_indices: List[int]
) -> torch.Tensor:
    """
    group 內的 softmax，若有多個正樣本，目標分佈是均勻的 multi-hot。
    回傳單一群組的 CE 樣本損失（標量）。
    """
    log_prob = torch.log_softmax(logits, dim=-1)
    tgt = torch.zeros_like(logits)
    if len(pos_indices) == 0:
        # 極少數異常情況（不應發生）：當作全負 → 取最大 logit 當 pseudo 正，避免 nan
        max_idx = int(torch.argmax(logits).item())
        pos_indices = [max_idx]
    w = 1.0 / float(len(pos_indices))
    for i in pos_indices:
        tgt[i] = w
    loss = -(tgt * log_prob).sum()
    return loss

def pairwise_margin_loss(
    logits: torch.Tensor,
    pos_indices: List[int],
    margin: float = 0.3
) -> torch.Tensor:
    """
    對每個正樣本 p 與每個負樣本 n：max(0, margin - (logit_p - logit_n))
    取平均。
    """
    if len(pos_indices) == 0:
        return torch.zeros((), device=logits.device)
    pos_logits = logits[pos_indices]                    # [P]
    neg_mask = torch.ones_like(logits, dtype=torch.bool)
    neg_mask[pos_indices] = False
    neg_logits = logits[neg_mask]                       # [N]
    if neg_logits.numel() == 0:
        return torch.zeros((), device=logits.device)
    # P x N
    diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
    loss = torch.relu(margin - diff).mean()
    return loss

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--mined_file", type=str, default="./data/reranker_mined_clean.jsonl")
    ap.add_argument("--output_dir", type=str, default="./models/reranker")
    ap.add_argument("--log_file", type=str, default="./logs/reranker_listwise.jsonl")
    ap.add_argument("--max_len", type=int, default=512)

    # Data/batching
    ap.add_argument("--batch_groups", type=int, default=8, help="一個 batch 多少個 query 的 group")
    ap.add_argument("--group_size", type=int, default=8, help="每組候選總數（正+負）")
    ap.add_argument("--max_pos_in_group", type=int, default=2, help="每組最多放幾個正樣本")
    ap.add_argument("--curriculum", type=str, default="", help='例如 "1x4,3x8" 代表前1個epoch用group_size=4，接著3個epoch用8')

    # Optim
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--warmup_ratio", type=float, default=0.2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # Pairwise margin aux
    ap.add_argument("--pair_margin", type=float, default=0.3)
    ap.add_argument("--lambda_pair", type=float, default=0.2)

    args = ap.parse_args([a for a in sys.argv[1:] if a.strip()])
    os.makedirs(args.output_dir, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # ---- Curriculum plan ----
    group_plan = parse_curriculum(args.curriculum, args.group_size, args.epochs)
    # 例：--curriculum "1x4,3x8" → [4,8,8,8]

    # ---- Data / Tokenizer ----
    base = BaseMined(args.mined_file, shuffle_pos=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    collate = make_collate(tokenizer, args.max_len)

    # ---- Model / Optim ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.base_ckpt, num_labels=1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 為了更準確 warmup，需要先估算總步數（考慮 curriculum 與 grad_accum）
    steps_per_epoch = []
    for gsz in group_plan:
        epoch_ds = EpochDataset(base, group_size=gsz, max_pos_in_group=args.max_pos_in_group)
        steps_per_epoch.append(math.floor(len(epoch_ds) / max(1, args.batch_groups)))
    total_opt_steps = sum(steps_per_epoch) // max(1, args.grad_accum)
    warmup_steps = int(total_opt_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    scaler = torch.amp.GradScaler(device="cuda", enabled=True)

    # ---- Train ----
    global_step = 0
    for ep in range(1, args.epochs + 1):
        # 每個 epoch 依 curriculum 重建 dataset/loader，動態重採樣負樣本
        cur_group_size = group_plan[ep - 1]
        epoch_ds = EpochDataset(base, group_size=cur_group_size, max_pos_in_group=args.max_pos_in_group)
        loader = DataLoader(epoch_ds, batch_size=args.batch_groups, shuffle=True,
                            drop_last=True, collate_fn=collate, pin_memory=True)

        model.train()
        t0 = time.time()
        tot_lw, tot_pw, seen = 0.0, 0.0, 0

        for step, (enc, slices, pos_lists) in enumerate(loader, start=1):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

            if (step - 1) % args.grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(**enc).logits.view(-1)   # [sum(group_sizes)]
                # 逐組計算 listwise CE 以及 pairwise margin
                start = 0
                lw_losses, pw_losses = [], []
                for (l, r), pos_idx_in_group in zip(slices, pos_lists):
                    g_log = logits[l:r]
                    lw = listwise_soft_ce_loss(g_log, pos_idx_in_group)
                    lw_losses.append(lw)

                    if args.lambda_pair > 0:
                        pw = pairwise_margin_loss(g_log, pos_idx_in_group, margin=args.pair_margin)
                    else:
                        pw = torch.zeros((), device=device)
                    pw_losses.append(pw)

                lw_loss = torch.stack(lw_losses).mean()
                pw_loss = torch.stack(pw_losses).mean()
                loss = (lw_loss + args.lambda_pair * pw_loss) / args.grad_accum

            scaler.scale(loss).backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if step % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

            tot_lw += lw_loss.item()
            tot_pw += pw_loss.item()
            seen += 1

        avg_lw = tot_lw / max(1, seen)
        avg_pw = tot_pw / max(1, seen)
        elapsed = time.time() - t0

        # log
        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": ep,
                "group_size": cur_group_size,
                "avg_listwise_loss": round(avg_lw, 6),
                "avg_pairwise_loss": round(avg_pw, 6),
                "optimizer_steps": global_step,
                "time_sec": round(elapsed, 2),
                "lr": scheduler.get_last_lr()[0],
            }) + "\n")

        print(f"[Listwise][Epoch {ep}] gsz={cur_group_size} "
              f"lw={avg_lw:.4f} pw={avg_pw:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
              f"time={elapsed:.1f}s")

    # ---- Save ----
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved reranker to {args.output_dir}")
    
if __name__ == "__main__":
    main()
