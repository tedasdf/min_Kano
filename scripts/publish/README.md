# Publishing Embed models

The uploader validates each final Sentence Transformers checkpoint before any
network operation. It creates one Hugging Face model repository per training
variant and is a dry run unless `--execute` is supplied.

Install the publishing dependency:

```bash
python -m pip install -e ".[publish]"
```

Provide a Hugging Face write token through the environment:

```bash
export HF_TOKEN="hf_..."
```

Do not put the token in a YAML file, command argument, repository, or SLURM
script committed to Git.

Validate all checkpoint folders without uploading:

```bash
python scripts/publish/upload_embed_models.py \
  --namespace YOUR_HF_USERNAME
```

Upload completed variants to private repositories:

```bash
python scripts/publish/upload_embed_models.py \
  --namespace YOUR_HF_USERNAME \
  --variants v1 v2 v3 \
  --private \
  --execute
```

Upload public repositories only after checking the model licence, model card,
training-data provenance, and evaluation results:

```bash
python scripts/publish/upload_embed_models.py \
  --namespace Sing0402 \
  --variants v1 v2 v3 \
  --no-private \
  --execute
```

Default repository names are:

```text
Sing0402/auslegal-embed-gte-inbatch
Sing0402/auslegal-embed-gte-bm25
Sing0402/auslegal-embed-gte-dense
Sing0402/auslegal-embed-gte-distilled
```

The script can also use credentials stored by `hf auth login` when `HF_TOKEN`
is not set.
