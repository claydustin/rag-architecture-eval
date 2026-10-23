"""Runs INSIDE a SageMaker Processing job.

SageMaker's ProcessingInput/Output channels move data between S3 and local
paths for us, so this script only ever touches local paths:

  input  channel "data"    -> /opt/ml/processing/input/data/{corpus,queries}.parquet
  input  channel "config"  -> /opt/ml/processing/input/config/<name>.yaml
  output channel "results" -> /opt/ml/processing/output/  (copied back to S3 on exit)

It wires those channels to the existing harness (rag_eval.run_eval.evaluate),
which reads data/*.parquet relative to the repo root — so we stage the input
parquet into that expected DATA_DIR, run evaluate(), and write CSVs to the
output dir. No boto3, no S3 calls here: the framework does the transfer.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from rag_eval.data import DATA_DIR
from rag_eval.run_eval import evaluate

IN_DATA = Path("/opt/ml/processing/input/data")
IN_CONFIG = Path("/opt/ml/processing/input/config")
OUT = Path("/opt/ml/processing/output")


def _stage_data() -> None:
    """Copy the input-channel parquet into the DATA_DIR the harness expects."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("corpus.parquet", "queries.parquet"):
        src = IN_DATA / name
        dst = DATA_DIR / name
        print(f"[proc] staging {src} -> {dst}")
        shutil.copyfile(src, dst)


def _load_config(config_name: str | None) -> dict:
    if config_name:
        path = IN_CONFIG / config_name
    else:
        # default: the only yaml in the config channel
        yamls = sorted(IN_CONFIG.glob("*.yaml"))
        if not yamls:
            raise SystemExit(f"no config yaml found in {IN_CONFIG}")
        path = yamls[0]
    print(f"[proc] config: {path}")
    return yaml.safe_load(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default=None,
                    help="yaml filename in the config channel (defaults to the only one)")
    a = ap.parse_args()

    _stage_data()
    cfg = _load_config(a.config_name)
    print(f"[proc] running: {cfg.get('name')}")

    summary, per_q = evaluate(cfg)

    OUT.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]
    per_q.to_csv(OUT / f"{name}_per_query.csv", index=False)
    # One summary row per run; SageMaker will sync the whole output dir to S3.
    (OUT / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(OUT / f"{name}_runs.csv", index=False)

    print(f"[proc] done: {name} — {summary['n_queries']} queries, "
          f"{summary['n_chunks']} chunks @ {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
