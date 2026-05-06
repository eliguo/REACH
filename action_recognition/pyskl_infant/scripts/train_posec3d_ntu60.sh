#!/bin/bash

PYTHON="/gpfs/scratch/yg3030/miniconda3/envs/pyskl/bin/python"

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=12345 \
  tools/train.py \
  configs/posec3d/c3d_light_ntu60_2d/joint.py \
  --launcher pytorch