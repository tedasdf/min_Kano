#!/usr/bin/env python3
"""Mine BM25 hard negatives from a retrieval split."""

from __future__ import annotations

import argparse
import json

from mini_kanon3.capabilities.embed.mining import mine_bm25_hard_negatives
from mini_kanon3.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embed/mine_bm25.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("method") != "bm25":
        parser.error("This command requires method: bm25")
    report = mine_bm25_hard_negatives(config)
    print(json.dumps(report["statistics"], indent=2))


if __name__ == "__main__":
    main()
