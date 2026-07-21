#!/usr/bin/env python3
"""Mine dense hard negatives using a trained Embed checkpoint."""

import argparse
import json

from mini_kanon3.capabilities.embed.mining import mine_dense_hard_negatives
from mini_kanon3.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embed/mine_dense.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("method") != "dense":
        parser.error("This command requires method: dense")
    print(json.dumps(mine_dense_hard_negatives(config)["statistics"], indent=2))


if __name__ == "__main__":
    main()
