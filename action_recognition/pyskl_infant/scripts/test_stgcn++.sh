#!/bin/bash

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=12356 \
  tools/test.py \
  configs/stgcn++/stgcn++_pyskl_infact2d_2dkp/j.py \
  -C work_dirs/stgcn++/stgcn++_pyskl_infact2d_withZ/j/latest.pth \
  --eval top_k_accuracy mean_class_accuracy \
  --launcher pytorch \
  > work_dirs/stgcn++/stgcn++_pyskl_infact2d_withZ/j/test_output.log 2>&1