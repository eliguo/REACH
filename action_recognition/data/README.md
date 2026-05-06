# data

Processing scripts for the InfActPrimitive and NTU RGB+D datasets. The actual data is not in the repo — see download links in `../README.md`.

## Layout

```
data/
├── add_z.py                          # 2D → 3D: append constant Z=1.0 channel
├── process_ntu2h36m_conversion.py    # NTU joint reorder to H36M layout
├── LABEL_INFO.txt                    # InfActPrimitive class labels
├── ntu/                              # NTU-specific build / convert scripts
│   ├── ntu120_3d_to_2d.py
│   ├── ntu120_build_3d_pkl.py        (+ .ipynb)
│   ├── ntu120_to_ntu60_2d.py
│   ├── ntu120_eda.ipynb
│   ├── show_skeleton_npy.py
│   └── txt2npy.py
└── eda/                              # exploratory notebooks
    ├── infact_eda.ipynb
    ├── output_eda.ipynb
    └── visualization_skeleton_layouts.ipynb
```

## Expected data layout (after download)

```
data/
├── InfActPrimitive/        # download from Google Drive (see ../README.md)
│   ├── 2d/
│   ├── 3d/
│   └── example.json
└── NTU/                    # download from official source
    ├── raw_npy120/
    ├── raw_txt120/
    ├── ntu60_2d.pkl
    ├── ntu120_2d.pkl
    └── ...
```

## Typical pipeline

For InfActPrimitive 2D → 3D (needed by GCN backbones expecting 3-channel input):

```bash
python add_z.py
```

For NTU raw → processed pkl:

```bash
cd ntu
python txt2npy.py
python ntu120_build_3d_pkl.py
python ntu120_3d_to_2d.py        # if a 2D variant is needed
```
