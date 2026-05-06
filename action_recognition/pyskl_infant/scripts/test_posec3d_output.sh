#!/bin/bash

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=12346 \
  tools/test.py \
  configs/posec3d/c3d_light_infact2d/joint_output.py \
  -C work_dirs/posec3d/c3d_light_infact2d/joint_output/latest.pth \
  --eval top_k_accuracy \
  --launcher pytorch \
  | tee work_dirs/posec3d/c3d_light_infact2d/joint_output/test_output.log
