#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --job-name="VAE_Synthetic_Claim_Filtering"
#SBATCH --partition=short
#SBATCH --time=0-04:00:00
#SBATCH --output=Experiments/logs/vae_filter_%j.out
#SBATCH --error=Experiments/logs/vae_filter_%j.err

mkdir -p Experiments/logs
mkdir -p Results/synthetic

module load python

cd "$SLURM_SUBMIT_DIR" || exit 1

source .venv/bin/activate

python3 -u Experiments/filter_lfm25_synthetic_claims.py \
  --input-csv Results/synthetic/raw_vae_best_total_2000_claims.csv \
  --output-csv Results/synthetic/filtered_vae_best_total_1000_claims.csv \
  --report-json Results/synthetic/filtered_vae_best_total_1000_report.json \
  --target-size 1000 \
  --near-duplicate-threshold 0.92

python3 -u Experiments/filter_lfm25_synthetic_claims.py \
  --input-csv Results/synthetic/raw_vae_best_reconstruction_2000_claims.csv \
  --output-csv Results/synthetic/filtered_vae_best_reconstruction_1000_claims.csv \
  --report-json Results/synthetic/filtered_vae_best_reconstruction_1000_report.json \
  --target-size 1000 \
  --near-duplicate-threshold 0.92
