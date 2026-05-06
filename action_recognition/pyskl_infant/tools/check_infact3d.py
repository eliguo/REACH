import pickle, numpy as np, sys
pkl = sys.argv[1]
d = pickle.load(open(pkl, "rb"))
assert "split" in d and "annotations" in d
a = d["annotations"][0]
print("keys:", a.keys())
kp = np.array(a["keypoint"])
print("kp shape:", kp.shape)
print("num classes? (from labels sample):", len(set(x["label"] for x in d["annotations"])))