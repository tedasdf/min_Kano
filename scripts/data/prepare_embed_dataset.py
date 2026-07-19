#!/usr/bin/env python3
"""Prepare the Embed retrieval dataset from a reproducible YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from mini_kanon3.config import load_config
from mini_kanon3.data.prepare_embed import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embed/prepare.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    prepare(
        input_path=Path(config["input"]),
        output=Path(config["output_root"]),
        seed=str(config["seed"]),
        ratios=tuple(float(value) for value in config["split_ratios"]),
        limits=[int(value) for value in config["context_limits"]],
    )


if __name__ == "__main__":
    main()
