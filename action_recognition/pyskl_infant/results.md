# Skeleton-based Action Recognition Model Benchmarking

## Dataset variants used

Three variants of the same annotation file used during experiments:

- **[2D (with score)](../Data/InfActPrimitive/2d/InfAct_plus.pkl)**  
  - `keypoint` shape per sample: `(T, V, 2)` (COCO-17 joints)  
  - `keypoint_score`: **present** (per-joint confidence)  
  - Pipelines that assume 2D must use `PreNormalize2D`, and components that reference `keypoint_score` will work.

- **[2D + Z (score kept)](../Data/InfActPrimitive/2d/InfAct_plus_withZ.pkl)**  
  - `keypoint` promoted to `(T, V, 3)` by adding a constant Z (we used **Z=+1.0**)  
  - `keypoint_score`: present  
  - Use **3D pipelines** (e.g., `PreNormalize3D`). Any code assuming 2D scores may need to skip score-related branches.

- **[2D + Z (no score)](../Data/InfActPrimitive/2d/InfAct_plus_withZ_noscore.pkl)**  
  - `keypoint` is `(T, V, 3)` with constant Z=1.0  
  - `keypoint_score`: **removed**  
  - Use **3D pipelines** (`PreNormalize3D`), **no** score-dependent transforms.

> **Rationale:** Some GCN backbones (e.g., CTRGCN, AAGCN, DGSTGCN, MSG3D) assume **3D input (C=3)** with `BatchNorm1d` dimensions `(M*V*C)`. Promoting 2D → 3D with a constant Z and setting `num_person=1` aligns tensor shapes with these backbones.

## Model Descriptions

### ST-GCN  
Spatial Temporal Graph Convolutional Network (ST-GCN) introduces the paradigm for modeling skeleton sequences as spatial–temporal graphs. Nodes represent joints and edges represent connectivity in space and time. Strong baseline but limited by fixed adjacency.

### ST-GCN++  
Enhances temporal modeling via dynamic edge importance and higher-order temporal convolutions. Did not outperform base version on this dataset, likely due to overfitting or under-utilized dynamics.

### AAGCN (Adaptive Attention GCN)  
Learns attention weights to dynamically adjust joint and edge importance. Underperforms in this setting due to noisy, irregular infant motions where learned attention generalizes poorly.

### CTR-GCN (Channel-wise Topology Refinement)  
Learns topology independently per channel, offering flexible edge structures. Moderate gains in Top-1 accuracy, but trails models with dynamic multi-scale designs.

### DG-STGCN (Dynamic Graph and Multi-Scale Temporal GCN)  
**Best performer**. Combines dynamic topology learning and multi-scale temporal convolutions to capture short- and long-term dependencies effectively.

### PoseC3D  
Uses 3D CNNs on voxelized joint data instead of GCNs. Performs competitively with DG-STGCN, showing 3D CNNs can effectively model spatiotemporal motion patterns.

### InfoGCN  
Aims to maximize mutual information between joints and motion, but suffered **representation collapse** — predictions converged to few dominant classes due to dataset imbalance. Accuracy oscillated across modes, consistent with original paper’s observation (~29.7% performance even with fine-tuning).

## Training & Evaluation Logic

### Environment
- Conda env: **`pyskl`**
- Single-GPU training/eval (deprecated launch utility):  
  ```bash
  python -m torch.distributed.launch --nproc_per_node=1 --master_port=<PORT> ... --launcher pytorch
  ```

### Training Loop
- **Optimizer:** SGD (`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`, `nesterov=True`)  
- **LR schedule:** CosineAnnealing (`by_epoch=False`, `min_lr=0`)  
- **Epochs:** 24  
- **Pipelines:**
  - 2D → `PreNormalize2D` → `GenSkeFeat(dataset='coco', feats=['j'])` → `UniformSample(clip_len=48)` → `PoseDecode` → `FormatGCNInput(num_person=1)`
  - 3D-promoted → `PreNormalize3D` (no score) → same sequence
- **Checkpoints:** saved per epoch under model-specific `work_dir`
- **Logs:** `TextLoggerHook(interval=100)`

### Evaluation
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12346 tools/test.py <CONFIG_PATH> -C <WORK_DIR>/latest.pth --eval top_k_accuracy mean_class_accuracy --launcher pytorch 2>&1 | tee <WORK_DIR>/test.log
```

## Results

| Model    | Top-1 Acc | Mean Class Acc |
|-----------|-----------:|---------------:|
| PoseC3D  | **0.6352** | 0.5527 |
| ST-GCN   | 0.5332 | 0.4163 |
| ST-GCN++ | 0.3597 | 0.4115 |
| CTR-GCN  | 0.4082 | 0.3994 |
| AAGCN   | 0.2985 | 0.3709 |
| DG-STGCN | **0.7449** | **0.5581** |
| MSG3D    | — | — |
| InfoGCN  | — | — |
