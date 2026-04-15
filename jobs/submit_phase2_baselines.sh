#!/bin/bash
#SBATCH --account=st-singha53-1-gpu
#SBATCH --job-name=if2rna_p2_submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --output=logs/phase2_submit_%j.out
#SBATCH --error=logs/phase2_submit_%j.err

set -euo pipefail

cd /scratch/st-singha53-1/schiluku/if2rna_new
mkdir -p logs

echo "Submitting Phase 2 ViS main run..."
vis_submit=$(sbatch jobs/train_if2rna_phase2_vis.sh)
vis_jobid=$(echo "$vis_submit" | awk '{print $4}')
echo "$vis_submit"

echo "Submitting Phase 2 ViT baseline run..."
vit_submit=$(sbatch jobs/train_if2rna_phase2_vit.sh)
vit_jobid=$(echo "$vit_submit" | awk '{print $4}')
echo "$vit_submit"

echo "Submitting Phase 2 mean-pooling baseline run..."
mean_submit=$(sbatch jobs/train_if2rna_phase2_meanpool.sh)
mean_jobid=$(echo "$mean_submit" | awk '{print $4}')
echo "$mean_submit"

echo "Submitting Phase 2 attention-MIL run..."
amil_submit=$(sbatch jobs/train_if2rna_phase2_attention_mil.sh)
amil_jobid=$(echo "$amil_submit" | awk '{print $4}')
echo "$amil_submit"

echo "Submitting Phase 2 Elastic-Net baseline run..."
enet_submit=$(sbatch jobs/train_if2rna_phase2_elastic_net.sh)
enet_jobid=$(echo "$enet_submit" | awk '{print $4}')
echo "$enet_submit"

echo "All baseline jobs submitted in parallel."
echo "  ViS job id:  $vis_jobid"
echo "  ViT job id:  $vit_jobid"
echo "  Mean job id: $mean_jobid"
echo "  AMIL job id: $amil_jobid"
echo "  ENet job id: $enet_jobid"
echo "Check queue with: squeue -u $USER"
