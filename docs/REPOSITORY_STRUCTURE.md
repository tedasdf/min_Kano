# Repository structure

Mini Kanon 3 separates reproducible configuration, immutable data, reusable implementation, thin commands, research decisions, reports, and large artifacts.

```text
configs/                    experiment and pipeline configuration
data/raw/                   immutable downloads with provenance/checksums
data/interim/               cleaned data, audits, positives, and negatives
data/processed/             final model-ready datasets
external/mleb/              external benchmark cache
research/<capability>/      decisions, evidence, and unresolved questions
notebooks/<capability>/     exploration only
scripts/{data,train,...}/   thin executable entry points
src/mini_kanon3/data/       shared data logic and leakage prevention
src/mini_kanon3/models/     shared backbone, pooling, registry, and outputs
src/mini_kanon3/capabilities/ independent single-task implementations
src/mini_kanon3/multitask/  deferred shared multi-task implementation
reports/                    human-readable results
artifacts/                  checkpoints, indexes, predictions, benchmarks
tests/{unit,integration,e2e}/ layered verification
```

Development order is **Embed → Classify → Enrich → Segment → Multitask**. Each final multi-task model must be compared against its corresponding single-task baseline.

## Embed data preparation

```powershell
python -m pip install -e .
python scripts/data/prepare_embed_dataset.py --config configs/embed/prepare.yaml
```

This creates a preserved positive-only dataset under `data/interim/embed/positive_only/` and model-ready `train/`, `validation/`, and `test/` retrieval objects under `data/processed/embed/`.
