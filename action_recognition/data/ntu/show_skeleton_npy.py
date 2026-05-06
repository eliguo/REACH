import numpy as np, os

PATH = "./raw_npy120/S001C001P001R001A001.skeleton.npy"

arr = np.load(PATH, allow_pickle=True)
d = arr.item() if isinstance(arr, np.ndarray) and arr.dtype == object else arr

print("file:", os.path.basename(PATH))
for k in sorted(d.keys()):
    v = d[k]
    if isinstance(v, np.ndarray):
        print(f"{k}: ndarray, shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"{k}: {type(v).__name__}, value={v}")
