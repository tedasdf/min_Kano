#!/usr/bin/env python3
"""Train Embed v4 using teacher relevance distributions."""

import argparse
from mini_kanon3.capabilities.embed.distillation import DistillationTrainer
from mini_kanon3.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embed/train_v4_distillation.yaml")
    args = parser.parse_args()
    report = DistillationTrainer(load_config(args.config)).train()
    print(f"Training complete. Final validation: {report['history'][-1]['validation']}")


if __name__ == "__main__": main()
