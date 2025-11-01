Here’s a clean English rewrite you can drop into your README.

# ADL_HW3_RAG

## Usage

* Install packages (Python 3.12):

```bash
pip install -r requirements.txt
```

* Save the corpus into a vector database (from `corpus.txt`):

```bash
python save_embeddings.py --retriever_model_path [your_model_path] --build_db
```

* Create `./.env` and put your own HF token inside:

```
hf_token="....."
```

(See the token docs: [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens))

* Inference:

```bash
python inference_batch.py \
  --test_data_path [your_data_path] \
  --retriever_model_path [your_retrieve_model_path] \
  --reranker_model_path [your_rerank_model_path] \
  --test_data_path ./data/test_open.txt
```

---

# ADL 2025 HW3 — Dense Retriever + Cross-Encoder Reranker

This project uses a two-stage retrieval pipeline:

* **Retriever** (bi-encoder, Sentence-Transformers): fast first-stage ranking
* **Reranker** (Cross-Encoder): precise re-ranking

It supports batch inference and simple visualization/analysis.

Contents

* Environment Setup
* Data Format & Folder Layout
* Train the Retriever
* Build Vector Index & Mine Hard Negatives
* Train the Reranker
* Batch Inference
* Evaluation & Plotting
* FAQ
* Quick Commands

---

## Environment Setup

* Recommended: Linux + Conda + NVIDIA GPU
* If you don’t use `requirements.txt`, install manually (example: CUDA 12.4):

```bash
conda create -n adl_hw3 python=3.10 -y
conda activate adl_hw3
python -m pip install -U pip

# Install PyTorch (replace cu124/cu121/cu118 or use cpu as needed)
pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.8.0"

# Other dependencies
pip install "transformers==4.56.1" "datasets==4.0.0" "tqdm==4.67.1" \
  "faiss-gpu-cu12==1.12.0" "sentence-transformers==5.1.0" \
  "python-dotenv==1.1.1" "accelerate==1.10.1" gdown matplotlib
```

* Or simply:

```bash
pip install -r requirements.txt
```

---

## Data Format & Folder Layout

Project structure (highlights)

```
hw3/
  data/
    corpus.txt               # Each line: pid<TAB>passage
    train.txt                # Each line: qid<TAB>query
    qrels.txt                # Each line: TREC style or qid pid rel
    reranker_mined.jsonl     # (optional) candidates/hard negatives mined by retriever
    reranker_mined_clean.jsonl
  models/
    retriever/               # Trained Sentence-Transformers model
    reranker/                # Trained Cross-Encoder model
  vector_database/
    passage_index.faiss      # FAISS index (built from corpus embeddings)
  logs/
    retriever_log.jsonl
    reranker_listwise.jsonl
  results/
    result.json              # Inference output (used by analysis scripts)
```

File formats

* `corpus.txt`: `pid<TAB>passage_text`
* `train.txt`: `qid<TAB>query_text`
* `qrels.txt`:

  * Supports TREC format: `qid Q0 pid rank score tag`
    or simplified: `qid pid rel` (any `rel > 0` is treated as relevant)
* `reranker_mined_*.jsonl` (produced by the mining script for reranker training), typical fields:

  * A record includes: `qid`, `query`, `positives`, `negatives` (or a `candidates` array with `doc_id` and `score`)

---

## Train the Retriever

* Model: Sentence-Transformers (e.g., `intfloat/multilingual-e5-small`)
* Loss: `MultipleNegativesRankingLoss` (MNRankLoss)
* Notes: `smart_batching_collate`, AMP, automatic device placement, per-epoch loss logging to `logs/retriever_log.jsonl`

Command

```bash
cd /home/cheweichang/Course/ADL/hw3
python3 train_retriever.py \
  --base_ckpt intfloat/multilingual-e5-small \
  --train_file ./data/train.txt \
  --corpus_file ./data/corpus.txt \
  --qrels_file ./data/qrels.txt \
  --output_dir ./models/retriever \
  --log_file ./logs/retriever_log.jsonl \
  --max_len 384 --batch_size 128 --lr 2e-5 --epochs 4 \
  --warmup_ratio 0.05 --grad_accum 1
```

Outputs

* Model: `models/retriever/`
* Log: `logs/retriever_log.jsonl` (with `avg_loss` per epoch)

---

## Build Vector Index & Mine Hard Negatives

1. Encode the corpus with the retriever and build a FAISS index

```bash
python3 save_embeddings.py \
  --model_path ./models/retriever \
  --corpus_file ./data/corpus.txt \
  --index_path ./vector_database/passage_index.faiss \
  --batch_size 256
```

2. Mine hard negatives for reranker training

```bash
# Check help to confirm argument names
python3 mine_hard_negatives_clean.py --help

# Example (adjust to your script’s exact args)
python3 mine_hard_negatives_clean.py \
  --retriever_model ./models/retriever \
  --corpus_file ./data/corpus.txt \
  --queries_file ./data/train.txt \
  --qrels_file ./data/qrels.txt \
  --faiss_index ./vector_database/passage_index.faiss \
  --topk 50 \
  --output ./data/reranker_mined.jsonl

# Optionally clean to a curated version
python3 mine_hard_negatives_clean.py \
  --input ./data/reranker_mined.jsonl \
  --output ./data/reranker_mined_clean.jsonl \
  --filter_duplicates --remove_trivial
```

Note: `data/reranker_mined_clean.jsonl` is included and can be used directly.

---

## Train the Reranker

* Model: Cross-Encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-12-v2`)
* Training: listwise CE (+ optional pairwise margin), group-based batching
* Key hyperparameters: `group_size` (candidates per query), `batch_groups` (groups per optimizer step)

Command

```bash
python3 train_reranker.py \
  --base_ckpt cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --mined_file ./data/reranker_mined_clean.jsonl \
  --output_dir ./models/reranker \
  --log_file ./logs/reranker_listwise.jsonl \
  --max_len 512 --batch_groups 8 --group_size 8 \
  --lr 1e-5 --epochs 10 --warmup_ratio 0.1 --grad_accum 2
```

Outputs

* Model: `models/reranker/`
* Log: `logs/reranker_listwise.jsonl` (with `avg_loss`)

---

## Batch Inference

* Script: `inference_batch.py`
* Flow: Retrieve (Retriever) → Re-rank (Reranker) → (Optional) Generate answer with an LLM (e.g., Qwen)
* Note: Use ASCII `--` for arguments (avoid the long en/em dash `–`). The script includes argv sanitization.

Command

```bash
# Optional: set HF token (if you need to download models)
export hf_token=YOUR_HF_TOKEN

python3 inference_batch.py \
  --data_folder ./data \
  --test_data_path ./data/test_open.txt \
  --qrels_path ./data/qrels.txt \
  --retriever_model_path ./models/retriever \
  --reranker_model_path ./models/reranker \
  --generator_model Qwen/Qwen3-1.7B \
  --result_file_name result.json
```

Output

* `results/result.json` (per-query candidates and scores; consumed by analysis scripts)

---

## Evaluation & Plotting

1. Overview analysis (loss curves, score distributions, Recall@k, candidate counts)

```bash
python3 analyze_inference.py --base_dir . --qrels data/qrels.txt
# Plots saved under results/plots/
```

2. Plot “one point per epoch” loss curves

```bash
python3 plot_avgloss.py --input logs/retriever_log.jsonl --output results/plots/loss_retriever_epoch.png --title "Retriever Avg Loss"
python3 plot_avgloss.py --input logs/reranker_listwise.jsonl --output results/plots/loss_reranker_epoch.png --title "Reranker Avg Loss"
```

Common artifacts

* `results/plots/loss_retriever.png`, `loss_reranker.png`
* `results/plots/<result>_score_hist.png`
* `results/plots/<result>_recall.png`
* `results/plots/<result>_cand_hist.png`
* `results/plots/summary.json`, `summary.md`

---

## FAQ

* **Argument errors (`unrecognized arguments`)**
  Usually caused by pasting a long dash (`–`). Replace with ASCII `--`, or rely on the script’s argv sanitizer (`inference_batch.py` supports this).

* **CUDA/CPU device mismatch**
  Ensure the training loop moves features/labels to `model.device` (the retriever training script already handles this).

* **AMP warnings (GradScaler/autocast)**
  Use `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` (the retriever has been updated accordingly).

* **FAISS installation issues**
  Match FAISS to your PyTorch CUDA version (e.g., `faiss-gpu-cu12` for CUDA 12.x).

---

## Quick Commands

```bash
# 1) Train retriever
python3 train_retriever.py --base_ckpt intfloat/multilingual-e5-small \
  --train_file data/train.txt --corpus_file data/corpus.txt --qrels_file data/qrels.txt \
  --output_dir models/retriever --log_file logs/retriever_log.jsonl \
  --max_len 384 --batch_size 128 --lr 2e-5 --epochs 4 --warmup_ratio 0.05

# 2) Build index
python3 save_embeddings.py --model_path models/retriever \
  --corpus_file data/corpus.txt --index_path vector_database/passage_index.faiss

# 3) Mine hard negatives (optional)
python3 mine_hard_negatives_clean.py --help

# 4) Train reranker
python3 train_reranker.py --base_ckpt cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --mined_file data/reranker_mined_clean.jsonl --output_dir models/reranker \
  --log_file logs/reranker_listwise.jsonl --max_len 512 --batch_groups 8 --group_size 8 \
  --lr 1e-5 --epochs 10 --warmup_ratio 0.1 --grad_accum 2

# 5) Batch inference
python3 inference_batch.py --data_folder ./data \
  --test_data_path ./data/test_open.txt --qrels_path ./data/qrels.txt \
  --retriever_model_path ./models/retriever --reranker_model_path ./models/reranker \
  --generator_model Qwen/Qwen3-1.7B --result_file_name result.json
```

> Note: I corrected obvious typos (e.g., `save_embbedings.py` → `save_embeddings.py`) and standardized dashes to ASCII `--`. If any script names/flags differ in your repo, keep your originals.
