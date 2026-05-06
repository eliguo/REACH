#!/bin/bash

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=12364 \
  tools/test.py \
  configs/ctrgcn/ctrgcn_pyskl_infact2d_2dkp/j.py \
  -C work_dirs/ctrgcn/ctrgcn_pyskl_infact2d_2dkp_withZ_singleM/j/latest.pth \
  --eval top_k_accuracy mean_class_accuracy \
  --launcher pytorch \
  2>&1 | tee work_dirs/ctrgcn/ctrgcn_pyskl_infact2d_2dkp_withZ_singleM/j/test.log