#!/usr/bin/env python3
"""Phase A artifact rebuild: regenerate semantic and sentence embeddings with PubMedBERT.

This script is an offline, one-shot rebuild. It sets PyTorch to use all available
CPU threads and runs the standard pipeline scripts with the v4 default model
(NeuML/pubmedbert-base-embeddings, 768-d).

Run from the repo root:
    python scripts/rebuild_pubmedbert_artifacts.py

Notes:
- HF_HOME should point outside the repository (e.g. C:\pubmed-hf-cache-v4) so the
  safety guard does not block model-cache writes.
- HNSW indexes cannot be built on Windows because hnswlib has no Windows wheel;
  build_hnsw_indexes.py is intended for Linux/Streamlit Cloud.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

# Use all physical cores for encoding; torch defaults to a conservative count.
torch.set_num_threads(os.cpu_count() or 8)
torch.set_num_interop_threads(os.cpu_count() or 8)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    """Run a command in the project root and stream output to stdout."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    logging.info("Running: %s", " ".join(command))
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=full_env,
        check=True,
    )
    elapsed = time.perf_counter() - start
    logging.info("Completed in %.1f seconds (exit=%s)", elapsed, result.returncode)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home:
        hf_home = str(Path(tempfile.gettempdir()) / "pubmedbert_hf_cache")
        os.environ["HF_HOME"] = hf_home
        logging.info("HF_HOME not set; using %s", hf_home)

    venv_python = PROJECT_ROOT / ".venv_win" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    # 1. Semantic embeddings (re-encodes existing semantic chunks with PubMedBERT).
    logging.info("=== Phase A rebuild: semantic embeddings ===")
    _run([str(venv_python), "-m", "src.embeddings"])

    # 2. Sentence index (regex sentence split + PubMedBERT embeddings).
    logging.info("=== Phase A rebuild: sentence index ===")
    _run(
        [
            str(venv_python),
            "scripts/build_indexes.py",
            "--strategies",
            "sentence",
            "--force",
        ]
    )

    logging.info("=== Phase A artifact rebuild complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
