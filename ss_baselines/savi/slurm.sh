#!/bin/bash
#SBATCH --job-name=isavi
#SBATCH --output=data/logs/%j.out
#SBATCH --error=data/logs/%j.err
#SBATCH --gres gpu:2
#SBATCH --nodes 2
#SBATCH --tasks-per-node 1
#SBATCH --time=72:00:00
#SBATCH --partition=clusterRTX


export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export MASTER_PORT=10001
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x


#srun python -u -m ss_baselines.savi.run \
#    --exp-config ss_baselines/savi/config/semantic_audionav/savi_pretraining_dialog_training.yaml \
#    --model-dir data/models/savi


python ss_baselines/savi/run.py --exp-config ss_baselines/savi/config/semantic_audionav/savi_interactive_1st_stage.yaml --model-dir data/models/AVLEN RL.SOFT_QUERY_REWARD True ALLOW_STOP True RL.QUERY_REWARD -1.2 RL.CONSECUTIVE_REWARD -0.5 REPLAY_STORE True
