#!/usr/bin/env python3
"""Score Embed v4 candidate passages with a cross-encoder teacher."""

import argparse
import json
from mini_kanon3.capabilities.embed.distillation import score_teacher_candidates
from mini_kanon3.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embed/score_teacher.yaml")
    args = parser.parse_args()
    report = score_teacher_candidates(load_config(args.config))
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
