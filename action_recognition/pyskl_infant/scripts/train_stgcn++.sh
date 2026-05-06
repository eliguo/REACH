#!/bin/bash

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=12355 \
  tools/train.py \
  configs/stgcn++/stgcn++_pyskl_infact2d_2dkp/j.py \
  --launcher pytorch