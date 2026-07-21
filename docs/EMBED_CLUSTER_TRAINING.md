# Embed cluster training

The SLURM pipeline runs v1 through v4 sequentially because each mining stage depends on the previous completed checkpoint. Training metrics are grouped in Weights & Biases under `mini-kanon3 / embed-v1-v4`.

## One-time environment setup

```bash
python -m pip install -e ".[train,tracking]"
wandb login
```

For non-interactive jobs, export `WANDB_API_KEY` before submission or store it through `wandb login`. Never commit the token.

## Submit

```bash
cd /fs04/scratch2/vf38/sloo0021/min_Kano
sbatch --export=ALL,PROJECT_DIR="$PWD",CONDA_ENV=minKvenv scripts/slurm/train_all_embed.slurm
```

Adjust the partition, GPU resource syntax, memory, and wall time in the `#SBATCH` header to match the cluster. Monitor with `squeue -u "$USER"` and `tail -f slurm-mini-kanon3-embed-<job-id>.out`.

Logged step metrics include training loss, learning rate, epoch/batch position, allocated/reserved GPU memory, and peak GPU memory. Logged epoch metrics include mean training loss, validation NDCG@10, Recall@1/5/10, MRR, and best-so-far validation summaries.

Set `WANDB_MODE=offline` before submission when compute nodes cannot reach the internet. Afterward run `wandb sync <offline-run-directory>` from a networked node.
