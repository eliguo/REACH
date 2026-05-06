# pyskl_infant

Our infant adaptation of [PySkl](https://github.com/kennymckormick/pyskl), based on commit `3f6795f`. This folder only contains the files we added or modified — to use it, clone PySkl and overlay these files.

## Setup

```bash
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
git checkout 3f6795f
# follow PySkl's install instructions

# overlay our changes
cp -r /path/to/pyskl_infant/configs/*    configs/
cp -r /path/to/pyskl_infant/tools/*      tools/
cp -r /path/to/pyskl_infant/scripts/*    .
cp -r /path/to/pyskl_infant/modified/*   .
```

## Layout

```
pyskl_infant/
├── configs/      # infant configs we added (8 backbones × InfActPrimitive)
├── scripts/      # train_*.sh / test_*.sh / .sbatch
├── tools/        # check_infact3d.py, make_infact_gif.py
└── modified/     # PySkl source files we changed
    └── changes.diff   # full diff against PySkl 3f6795f
```

## Configs

| Backbone   | Config path                                                  |
|------------|--------------------------------------------------------------|
| ST-GCN     | `configs/stgcn/stgcn_pyskl_infact2d_2dkp/j.py`               |
| ST-GCN++   | `configs/stgcn++/stgcn++_pyskl_infact2d_2dkp/j.py`           |
| CTR-GCN    | `configs/ctrgcn/ctrgcn_pyskl_infact2d_2dkp/j.py`             |
| AAGCN      | `configs/aagcn/aagcn_pyskl_infact2d_2dkp/j.py`               |
| DG-STGCN   | `configs/dgstgcn/infact_primitive_2dkp/j.py`                 |
| PoseC3D    | `configs/posec3d/c3d_light_infact2d/joint.py`                |

NTU 2D variants for PoseC3D are also included under `configs/posec3d/c3d_light_ntu60_2d/` and `c3d_light_ntu120_2d/`.

## Modified files

Three files in PySkl proper were changed (small tweaks for infant training):

- `configs/posec3d/c3d_light_ntu60_xsub/joint.py` — minor
- `configs/posec3d/slowonly_r50_ucf101_k400p/s1_joint.py` — `times=10→1`, `lr=0.01→0.001`
- `demo/demo_skeleton.py` — visualization tweaks

See `modified/changes.diff` for the full diff.

## Running

After overlaying, from the PySkl root:

```bash
bash train_dgstgcn.sh    # train DG-STGCN on InfActPrimitive
bash test_dgstgcn.sh     # evaluate
```

Edit data paths in the configs before running.
