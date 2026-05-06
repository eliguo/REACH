#!/usr/bin/env python
# coding: utf-8

"""
Build NTU RGB+D 120 3D skeleton annotation PKL with one-shot split.

- Input:  all *.skeleton.npy in RAW_DIR (dict with skel_body0, etc.)
- Output: OUT_PKL, with structure:
    {
      "split": {
          "train":    [...],  # auxiliary 100 classes
          "exemplar": [...],  # 20 exemplar sequences
          "test":     [...]   # remaining samples of 20 novel classes
      },
      "annotations": [ {...}, {...}, ... ]
    }
"""

import os
import re
import pickle
import numpy as np

RAW_DIR = "./raw_npy120"
OUT_PKL = "ntu120_3d.pkl"

# filename pattern: SxxxCxxxPxxxRxxxAyyy.skeleton.npy
FNAME_RE = re.compile(r"^(S\d{3}C\d{3}P\d{3}R\d{3}A(\d{3}))\.skeleton\.npy$")

# 20 novel classes (A1, A7, ..., A115)
NOVEL_CLASSES = {
    1, 7, 13, 19, 25,
    31, 37, 43, 49, 55,
    61, 67, 73, 79, 85,
    91, 97, 103, 109, 115,
}

# 20 exemplar sequences (exact IDs, without .skeleton.npy)
EXEMPLARS = {
    "S001C003P008R001A001",
    "S001C003P008R001A007",
    "S001C003P008R001A013",
    "S001C003P008R001A019",
    "S001C003P008R001A025",
    "S001C003P008R001A031",
    "S001C003P008R001A037",
    "S001C003P008R001A043",
    "S001C003P008R001A049",
    "S001C003P008R001A055",
    "S018C003P008R001A061",
    "S018C003P008R001A067",
    "S018C003P008R001A073",
    "S018C003P008R001A079",
    "S018C003P008R001A085",
    "S018C003P008R001A091",
    "S018C003P008R001A097",
    "S018C003P008R001A103",
    "S018C003P008R001A109",
    "S018C003P008R001A115",
}


def load_dict(path: str):
    """Load one *.skeleton.npy as dict."""
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return arr.item()
    return arr


def build_ann(npy_path: str) -> dict:
    """Build one annotation entry from a skeleton npy file."""
    base = os.path.basename(npy_path)
    m = FNAME_RE.match(base)
    if m is None:
        raise ValueError(f"Unexpected filename: {base}")
    seq_id = m.group(1)         # SxxxCxxxPxxxRxxxAyyy
    label_id = int(m.group(2))  # yyy → int

    d = load_dict(npy_path)

    xyz = d["skel_body0"].astype("float16")   # (T, V, 3)
    T, V, _ = xyz.shape

    keypoint = xyz[None, ...]                                 # (1, T, V, 3), float16
    keypoint_score = np.ones((1, T, V, 1), dtype="float16")   # (1, T, V, 1), float16

    ann = {
        "start_frame": 0,
        "end_frame": int(T - 1),
        "pos_label": None,
        "frame_dir": seq_id,
        "img_shape": [1080, 1920],
        "original_shape": [1080, 1920],
        "total_frames": int(T),

        "keypoint": keypoint,
        "keypoint_score": keypoint_score,

        "source": "ntu",
        "label": label_id,
    }
    return ann


def main():
    files = sorted(
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".skeleton.npy")
    )
    print("num skeleton npy files:", len(files))

    annotations = []
    skipped = []

    for i, fname in enumerate(files):
        npy_path = os.path.join(RAW_DIR, fname)
        if i % 500 == 0:
            print(f"{i}/{len(files)}")

        try:
            ann = build_ann(npy_path)
            annotations.append(ann)
        except Exception as e:
            print("skip:", fname, "reason:", e)
            skipped.append(fname)

    print("built annotations:", len(annotations))
    if skipped:
        print("skipped files:", len(skipped))

    # one-shot split
    train_list = []
    exemplar_list = []
    test_list = []

    for ann in annotations:
        label = ann["label"]
        seq = ann["frame_dir"]

        if label not in NOVEL_CLASSES:
            train_list.append(seq)
        else:
            if seq in EXEMPLARS:
                exemplar_list.append(seq)
            else:
                test_list.append(seq)

    split = {
        "train": train_list,
        "exemplar": exemplar_list,
        "test": test_list,
    }

    final_obj = {
        "split": split,
        "annotations": annotations,
    }

    with open(OUT_PKL, "wb") as f:
        pickle.dump(final_obj, f, protocol=4)

    print("saved:", OUT_PKL)
    print("train:", len(train_list),
          "exemplar:", len(exemplar_list),
          "test:", len(test_list))


if __name__ == "__main__":
    main()