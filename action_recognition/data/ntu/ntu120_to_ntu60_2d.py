#!/usr/bin/env python
# coding: utf-8

"""
Build a smaller NTU-60-like 2D skeleton PKL from ntu120_2d.pkl,
by keeping only subjects S001–S017.

Input:
    IN_PKL = "ntu120_2d.pkl"

Output:
    OUT_PKL = "ntu60_2d.pkl"
    Same structure:
      {
        "split": {
            "train": [...],
            "exemplar": [...],
            "test": [...]
        },
        "annotations": [...]
      }
    but only sequences whose frame_dir starts with S001..S017.
"""

import os
import pickle

IN_PKL = "ntu120_2d.pkl"
OUT_PKL = "ntu60_2d.pkl"

def main():
    assert os.path.exists(IN_PKL), f"Input PKL not found: {IN_PKL}"

    with open(IN_PKL, "rb") as f:
        obj = pickle.load(f)

    split = obj["split"]
    anns = obj["annotations"]

    # keep subjects S001..S017
    def is_kept(frame_dir: str) -> bool:
        # frame_dir like "S001C001P001R001A001"
        sid = int(frame_dir[1:4])
        return 1 <= sid <= 17

    new_anns = [ann for ann in anns if is_kept(ann["frame_dir"])]
    kept_ids = {ann["frame_dir"] for ann in new_anns}

    def filt(id_list):
        return [x for x in id_list if x in kept_ids]

    new_split = {}
    for k, v in split.items():
        new_split[k] = filt(v)

    out_obj = {
        "split": new_split,
        "annotations": new_anns,
    }

    with open(OUT_PKL, "wb") as f:
        pickle.dump(out_obj, f, protocol=4)

    print(f"Loaded: {IN_PKL}")
    print(f"Total annotations (orig): {len(anns)}")
    print(f"Total annotations (S001–S017): {len(new_anns)}")
    print("Split sizes (new):", {k: len(v) for k, v in new_split.items()})
    print(f"Saved: {OUT_PKL}")

if __name__ == "__main__":
    main()