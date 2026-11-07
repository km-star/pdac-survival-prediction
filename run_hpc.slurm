#!/bin/bash
#SBATCH --job-name=pdac
#SBATCH --partition=CHANGE_ME        # a CPU partition; this job does not use a GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=pdac_%x_%j.log

# One script, three runs. Pick with RUN=loo | rskf | null (default loo):
#   sbatch run_hpc.slurm                            # leave-one-out, 5 seeds   (~5 h on 32 cores)
#   RUN=rskf sbatch run_hpc.slurm                   # repeated 5-fold x 10      (~2 h)
#   RUN=null sbatch run_hpc.slurm                   # permutation null, 20 shuffles (~20 h; resubmit to resume)
# Resubmitting the same job resumes from OUT/preds/. Nothing is recomputed.
set -euo pipefail
source ~/.bashrc
conda activate pdac

RUN=${RUN:-loo}
DATA=$HOME/pdac_project/43018_2023_697_MOESM3_ESM.xlsx
RES=$HOME/pdac_results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"

case "$RUN" in
  loo)  python -u pdac_pipeline.py --data "$DATA" --outdir "$RES/loo" --parts all \
          --seeds 42 43 44 45 46 --n-jobs "$SLURM_CPUS_PER_TASK" ;;
  rskf) python -u pdac_pipeline.py --data "$DATA" --outdir "$RES/rskf" --parts all \
          --cv rskf --cv-folds 5 --cv-repeats 10 --n-jobs "$SLURM_CPUS_PER_TASK" ;;
  null) python -u pdac_pipeline.py --data "$DATA" --outdir "$RES/loo" --parts nulltest \
          --null-perms 20 --n-jobs "$SLURM_CPUS_PER_TASK" ;;
  *)    echo "RUN must be loo, rskf or null"; exit 2 ;;
esac
