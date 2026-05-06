# Action Recognition

Skeleton-based action recognition benchmarks on the **InfActPrimitive** infant dataset and **NTU RGB+D** adult dataset, plus an explored **InfoGCN** baseline.

The main results in the report come from the PySkl pipeline. InfoGCN was an early baseline we tried but did not include in the final benchmark table.

## Layout

```
action_recognition/
├── data/                 # data processing scripts (no actual data — see below)
│   ├── add_z.py                          # 2D → 3D by appending Z=1.0 channel
│   ├── process_ntu2h36m_conversion.py    # NTU joint reordering to H36M layout
│   ├── ntu/                              # NTU-specific build / convert scripts
│   └── eda/                              # exploratory notebooks
├── infogcn/              # explored InfoGCN baseline (not in final results)
├── pyskl_infant/         # our infant adaptation of PySkl — see its README
├── notebooks/            # run_experiments.ipynb, read_results.ipynb
└── assets/               # demo gifs
```

## Datasets

Datasets are not in the repo. Download them yourself.

**InfActPrimitive** — preprocessed 2D and 3D infant skeleton data:
https://drive.google.com/file/d/1TiuTul5b5XtJgKZeOCnrAH8WKmxb6Rld/view?usp=sharing

Place under `data/InfActPrimitive/`.

**NTU RGB+D** — full dataset from the official source:
https://rose1.ntu.edu.sg/dataset/actionRecognition/

Skeleton-only mirrors:
- https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H
- https://drive.google.com/open?id=1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w

Place under `data/NTU/`.

## Reproducing results

PySkl benchmarks (the main numbers in the report): see `pyskl_infant/README.md`.

InfoGCN baseline: `cd infogcn && bash train_infact2d.sh`.
