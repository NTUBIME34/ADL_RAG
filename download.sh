#!/usr/bin/env bash
# ADL2025 HW3 - download.sh (single archive version)
# Purpose: Download ONE archive that contains model/retriever and model/reranker, then extract to ./models/
# No pip installs; download + extract only. Enforce total size <= 2GB.

set -euo pipefail

### ======== [1] USER FILL SECTION ========
# 二擇一填即可：Drive 檔案 ID 或可直連的 HTTPS 下載連結
MODEL_DRIVE_ID="1zqrG95OZIsfpmutn2ExJPlwYizpxsf9d"         # e.g. 1AbCdefGhIjkLMNOP-xxxx
MODEL_URL="__OPTIONAL_HTTPS_URL__"   # e.g. https://storage.googleapis.com/... (pre-signed)

### ======== [2] CONSTANTS / PATHS ========
ROOT_DIR="$(pwd)"
MODELS_DIR="${ROOT_DIR}/models"
RETRIEVER_DIR="${MODELS_DIR}/retriever"
RERANKER_DIR="${MODELS_DIR}/reranker"
WORK_DIR="${ROOT_DIR}/.download_tmp"
PKG_PATH="${WORK_DIR}/model_pkg.bin"
EXTRACT_DIR="${WORK_DIR}/extract"

MAX_BYTES=$((2 * 1024 * 1024 * 1024))  # 2GB

### ======== [3] HELPERS ========
log() { echo "[download.sh] $*" >&2; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { log "Missing required tool: $1"; exit 1; }
}

download_with_curl() {
  local url="$1" out="$2"
  need_cmd curl
  curl -L --fail --retry 3 --connect-timeout 15 --max-time 1800 -o "${out}" "${url}"
}

download_with_gdown_id() {
  local file_id="$1" out="$2"
  need_cmd gdown
  gdown --fuzzy "https://drive.google.com/uc?id=${file_id}" -O "${out}"
}

detect_and_extract() {
  local pkg="$1" target_dir="$2"
  mkdir -p "${target_dir}"

  local mime
  mime="$(file -b --mime-type "${pkg}" || true)"

  if [[ "${pkg}" == *.tar.gz || "${pkg}" == *.tgz || "${mime}" == "application/x-gzip" ]]; then
    tar -xzf "${pkg}" -C "${target_dir}"
  elif [[ "${pkg}" == *.zip || "${mime}" == "application/zip" ]]; then
    need_cmd unzip
    unzip -q "${pkg}" -d "${target_dir}"
  else
    log "Unknown package format. Please upload .tar.gz/.tgz or .zip."
    exit 1
  fi
}

ensure_nonempty_dir() {
  local d="$1"
  [[ -d "${d}" ]] || { log "Directory not found: ${d}"; exit 1; }
  [[ -n "$(ls -A "${d}" 2>/dev/null || true)" ]] || { log "Directory is empty: ${d}"; exit 1; }
}

check_total_size_le_2g() {
  local total
  total=$(du -sb "${MODELS_DIR}" | awk '{print $1}')
  if (( total > MAX_BYTES )); then
    log "Models exceed 2GB limit (current: ${total} bytes). Please reduce size."
    exit 1
  fi
}

move_models_from_source() {
  local src_base="$1"  # 可能是 extract/model 或 extract（若壓縮內直接是 retriever/、reranker/）
  local src_ret="${src_base}/retriever"
  local src_rer="${src_base}/reranker"

  [[ -d "${src_ret}" && -d "${src_rer}" ]] || {
    log "Expected subfolders 'retriever' and 'reranker' under: ${src_base}"
    log "Archive should be: models/retriever and models/reranker (or top-level retriever & reranker)."
    exit 1
  }

  mkdir -p "${MODELS_DIR}"
  rm -rf "${RETRIEVER_DIR}" "${RERANKER_DIR}"
  mv "${src_ret}" "${RETRIEVER_DIR}"
  mv "${src_rer}" "${RERANKER_DIR}"

  ensure_nonempty_dir "${RETRIEVER_DIR}"
  ensure_nonempty_dir "${RERANKER_DIR}"
}

### ======== [4] MAIN ========
main() {
  log "Start downloading single model archive..."
  mkdir -p "${WORK_DIR}" "${EXTRACT_DIR}"

  # Download one archive
  if [[ "${MODEL_URL}" != "__OPTIONAL_HTTPS_URL__" && -n "${MODEL_URL}" ]]; then
    download_with_curl "${MODEL_URL}" "${PKG_PATH}"
  elif [[ "${MODEL_DRIVE_ID}" != "__FILL_ME__" && -n "${MODEL_DRIVE_ID}" ]]; then
    download_with_gdown_id "${MODEL_DRIVE_ID}" "${PKG_PATH}"
  else
    log "Please set MODEL_URL (HTTPS) or MODEL_DRIVE_ID (Google Drive file ID)."
    exit 1
  fi

  # Extract
  log "Extracting archive..."
  detect_and_extract "${PKG_PATH}" "${EXTRACT_DIR}"

  # Locate base that contains retriever/ and reranker/
  # 優先找 extract/model，其次允許壓縮內直接是 retriever/、reranker/
  local base=""
  if [[ -d "${EXTRACT_DIR}/models" ]]; then
    base="${EXTRACT_DIR}/models"
  else
    base="${EXTRACT_DIR}"
  fi

  move_models_from_source "${base}"

  # Size check
  check_total_size_le_2g

  # Clean & summary
  rm -rf "${WORK_DIR}"
  log "Done. Models placed under ./models/"
  log "Tree:"
  (cd "${MODELS_DIR}" && find . -maxdepth 2 -type d | sort)
}

main "$@"
