#!/bin/bash

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=12347 \
  tools/train.py \
  configs/aagcn/aagcn_pyskl_infact2d_2dkp/j.py \
  --launcher pytorch