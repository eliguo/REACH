#!/bin/bash

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=12345 \
  tools/train.py \
  configs/posec3d/c3d_light_infact2d/joint.py \
  --launcher pytorch