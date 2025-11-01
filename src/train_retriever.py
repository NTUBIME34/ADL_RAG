# train_retriever.py — manual loop + epoch logs
import os, time, json, argparse
from pathlib import Path
from typing import List
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer, InputExample, losses
from data_utils import load_corpus, load_qrels, load_train, build_retriever_pairs

def make_model(base_ckpt: str, max_len: int) -> SentenceTransformer:
    model = SentenceTransformer(base_ckpt, trust_remote_code=True)
    for m in model.modules():
        if hasattr(m, "max_seq_length"):
            m.max_seq_length = max_len
    model.max_seq_length = max_len
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", type=str, default="intfloat/multilingual-e5-small")
    parser.add_argument("--train_file", type=str, default="./data/train.txt")
    parser.add_argument("--corpus_file", type=str, default="./data/corpus.txt")
    parser.add_argument("--qrels_file", type=str, default="./data/qrels.txt")
    parser.add_argument("--output_dir", type=str, default="./models/retriever")
    parser.add_argument("--log_file", type=str, default="./logs/retriever_log.jsonl")
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # ---------- data ----------
    corpus = load_corpus(args.corpus_file)
    qrels = load_qrels(args.qrels_file)
    train_items = load_train(args.train_file)
    pairs = build_retriever_pairs(train_items, corpus, qrels, prefer_qrels=True)

    examples: List[InputExample] = []
    for q, p in pairs:
        # MultipleNegativesRankingLoss 期望每個樣本為 [anchor, positive]
        examples.append(InputExample(texts=[f"query: {q}", f"passage: {p}"]))

    model = make_model(args.base_ckpt, args.max_len)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    train_loader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=model.smart_batching_collate,  # 要用 ST 的 smart batching
        pin_memory=True,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    setattr(model, "_target_device", device)  # 讓 ST 內部模組知道目標裝置

    total_steps = (len(train_loader) * args.epochs) // max(1, args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 使用新版 AMP 介面
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, seen_batches = 0.0, 0
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            if (step - 1) % args.grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            # smart_batching_collate 回傳 (features, labels)
            features, labels = batch  # labels 通常為 None（MNRankLoss 不需要）
            # 搬移到與模型相同裝置
            features = [
                {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in feat.items()}
                for feat in features
            ]
            if torch.is_tensor(labels):
                labels = labels.to(device, non_blocking=True)

            # 保險檢查：MNRankLoss 需要 2 個句子（anchor, positive）
            assert isinstance(features, list) and len(features) == 2, \
                f"Expected 2 features (anchor,pos), got {len(features)}"

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                step_loss = loss_fn(features, labels)  # 單步原始 loss
                loss = step_loss / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                if args.clip_norm and args.clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

            # 正確累積 epoch loss（用未縮放的 step_loss）
            epoch_loss += float(step_loss.detach().cpu())
            seen_batches += 1

            # 週期性列印訓練進度
            if args.log_steps > 0 and (step % args.log_steps == 0):
                cur_lr = scheduler.get_last_lr()[0]
                print(f"[Epoch {epoch} Step {step}/{len(train_loader)}] "
                      f"loss={float(step_loss):.4f} lr={cur_lr:.2e}")

        avg_loss = epoch_loss / max(1, seen_batches)
        elapsed = time.time() - t0
        last_lr = scheduler.get_last_lr()[0]

        # ----- write epoch log -----
        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "avg_loss": round(avg_loss, 6),
                "optimizer_steps": global_step,
                "lr": last_lr,
                "time_sec": round(elapsed, 2),
                "num_batches": seen_batches,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "max_len": args.max_len
            }) + "\n")

        print(f"[Retriever][Epoch {epoch}] loss={avg_loss:.4f} lr={last_lr:.2e} time={elapsed:.1f}s")

    model.save(args.output_dir)
    print(f"[Retriever] Saved to {args.output_dir} ; logs -> {args.log_file}")

if __name__ == "__main__":
    main()
