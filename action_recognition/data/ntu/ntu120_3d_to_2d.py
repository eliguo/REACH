#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

IN_PKL = "ntu120_3d.pkl"
OUT_PKL = "ntu120_2d.pkl"


def main():
    print("Loading:", IN_PKL)
    with open(IN_PKL, "rb") as f:
        obj = pickle.load(f)

    split = obj["split"]
    anns = obj["annotations"]
    print("Num annotations:", len(anns))

    new_anns = []
    for i, ann in enumerate(anns):
        if i % 5000 == 0:
            print(f"{i}/{len(anns)}")

        kp3d = ann["keypoint"]          # list or ndarray, shape (1, T, 25, 3)
        kp3d_arr = np.array(kp3d)

        # drop z dimension → keep x,y only
        kp2d = kp3d_arr[..., :2].astype("float16")    # (1, T, 25, 2)

        # keypoint_score shape stays the same
        ks = np.array(ann["keypoint_score"]).astype("float16")

        new_ann = dict(ann)
        new_ann["keypoint"] = kp2d
        new_ann["keypoint_score"] = ks
        new_anns.append(new_ann)

    out_obj = {
        "split": split,
        "annotations": new_anns,
    }

    with open(OUT_PKL, "wb") as f:
        pickle.dump(out_obj, f, protocol=4)

    print("Saved:", OUT_PKL)
    print("split sizes:", {k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()