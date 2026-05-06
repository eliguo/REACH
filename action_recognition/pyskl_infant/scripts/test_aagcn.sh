#!/bin/bash

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=12348 \
  tools/test.py \
  configs/aagcn/aagcn_pyskl_infact2d_2dkp/j.py \
  -C work_dirs/aagcn/aagcn_pyskl_infact2d_2dkp/j/latest.pth \
  --eval top_k_accuracy mean_class_accuracy \
  --launcher pytorch \
  > work_dirs/aagcn/aagcn_pyskl_infact2d_2dkp/j/test_output.log 2>&1