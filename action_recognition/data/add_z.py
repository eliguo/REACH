#!/usr/bin/env python3
import pickle
import numpy as np
import os
import argparse

def add_z(kp, z=0.0):
    kp = np.array(kp)
    if kp.shape[-1] == 2:
        zc = np.full((*kp.shape[:-1], 1), z, dtype=kp.dtype)
        return np.concatenate([kp, zc], axis=-1)
    return kp

def main():
    ap = argparse.ArgumentParser(description="Add constant Z channel to 2D keypoints and remove keypoint_score")
    ap.add_argument("--in",  dest="inp",  required=True, help="input pkl path")
    ap.add_argument("--out", dest="outp", required=True, help="output pkl path")
    ap.add_argument("--z",   type=float,  default=1.0,   help="Z value to add (default=1.0)")
    args = ap.parse_args()

    with open(args.inp, "rb") as f:
        data = pickle.load(f)

    cnt = 0
    for item in data.get("annotations", []):
        if "keypoint" in item:
            item["keypoint"] = add_z(item["keypoint"], z=args.z)
            if "keypoint_score" in item:
                item.pop("keypoint_score")
            cnt += 1

    os.makedirs(os.path.dirname(args.outp), exist_ok=True)
    with open(args.outp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Processed {cnt} samples (added Z={args.z}, removed keypoint_score).")
    print(f"Saved to: {os.path.abspath(args.outp)}")

if __name__ == "__main__":
    main()