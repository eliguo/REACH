#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
import numpy as np
import torch

# headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio
from mmcv import Config
from mmcv.runner import load_checkpoint


# --------------------------
# Args
# --------------------------
def parse_args():
    p = argparse.ArgumentParser("InfActPrimitive GIF (pyskl DG-STGCN) with stability + diagnostics")

    p.add_argument("--repo_root", type=str, default=None,
                   help="path to repo root (Video-Based-Infant-Action-Recognition)")
    p.add_argument("--config", type=str, default=None, help="pyskl config path")
    p.add_argument("--checkpoint", type=str, default=None, help="checkpoint .pth")
    p.add_argument("--pkl", type=str, required=True, help="InfAct pkl path")

    # choose ONE of these:
    p.add_argument("--frame_dir", type=str, default=None,
                   help="render a single segment by exact frame_dir")
    p.add_argument("--group_prefix", type=str, default=None,
                   help="stitch all segments whose frame_dir startswith this prefix, e.g. YouTube_001 or D02_1")

    # if you don't provide frame_dir / group_prefix, pick from split list
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--index", type=int, default=0, help="which sample in split list")

    # stitching behavior
    p.add_argument("--gap_policy", type=str, default="skip", choices=["pad", "skip"],
                   help="pad: fill missing frames by last valid pose; skip: drop missing frames")
    p.add_argument("--boundary_pad", type=int, default=0,
                   help="add N frames padding on both sides of the global timeline (optional)")

    # inference/render
    p.add_argument("--clip_len", type=int, default=48)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=5000)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=str, default="infact_demo.gif")
    p.add_argument("--device", type=str, default="cuda:0")

    # stability / diagnostics
    p.add_argument("--show_topk", type=int, default=2, help="show top-k probs on title (recommended 2 or 3)")
    p.add_argument("--uncertain_margin", type=float, default=0.05,
                   help="if (top1-top2) < margin, mark as UNCERTAIN and use gray")
    p.add_argument("--pred_vote", type=int, default=0,
                   help="majority vote window size on predictions for display (0=off, 3 or 5 recommended)")
    p.add_argument("--kp_smooth", type=int, default=3,
                   help="temporal smoothing window on keypoints before inference (0=off, 3 or 5 recommended)")
    p.add_argument("--kp_smooth_mode", type=str, default="median", choices=["median", "mean"],
                   help="median is more robust to outliers")
    p.add_argument("--debug_every", type=int, default=25,
                   help="print diagnostics every N rendered frames (0 disables)")
    
    p.add_argument("--youtube_start", type=int, default=None,
                   help="start YouTube ID, e.g. 1 for YouTube_001")
    p.add_argument("--youtube_end", type=int, default=None,
                   help="end YouTube ID, e.g. 5 for YouTube_005")
    p.add_argument("--seconds_per_segment", type=float, default=None,
                   help="if set, keep only the first N seconds from each matched segment")
    p.add_argument("--source_fps", type=float, default=None,
                   help="source FPS used to convert seconds_per_segment to frame count")

    return p.parse_args()


# --------------------------
# Constants
# --------------------------
COCO_INWARD = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
    (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
    (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)
]

LABEL_NAMES = ["Supine", "Prone", "Sitting", "Standing", "All-fours"]
LABEL_COLORS = {
    0: (0.20, 0.55, 0.90),
    1: (0.90, 0.35, 0.25),
    2: (0.25, 0.70, 0.35),
    3: (0.60, 0.40, 0.85),
    4: (0.95, 0.75, 0.20),
}
COLOR_GRAY = (0.50, 0.50, 0.50)


# --------------------------
# Utils
# --------------------------
def draw_skeleton_2d(ax, kp_2d, color, lw=2, s=18):
    # kp_2d: (V, C>=2)
    kp_2d = kp_2d[:, :2]
    xs = kp_2d[:, 0]
    ys = kp_2d[:, 1]
    ax.scatter(xs, ys, c=[color], s=s)
    for a, b in COCO_INWARD:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], c=color, lw=lw)
    ax.invert_yaxis()
    ax.set_axis_off()


def window_at_t(kp_1TVC, t, clip_len):
    T = kp_1TVC.shape[1]
    start = t - (clip_len - 1)
    idxs = np.arange(start, t + 1)
    idxs = np.clip(idxs, 0, T - 1)
    return kp_1TVC[:, idxs, :, :]  # (1, clip_len, V, C)


def infer_repo_root(args):
    if args.repo_root is not None:
        return os.path.abspath(args.repo_root)
    # infer from this file location: <repo_root>/pyskl/tools/make_infact_gif.py
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../pyskl/tools
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def temporal_smooth_kp(kp_1TVC, k=3, mode="median"):
    """
    kp_1TVC: (1, T, V, C), can contain NaNs.
    Apply smoothing along T for each (V,C) independently.
    """
    if k <= 1:
        return kp_1TVC
    assert kp_1TVC.ndim == 4 and kp_1TVC.shape[0] == 1
    T = kp_1TVC.shape[1]
    pad = k // 2
    out = kp_1TVC.copy()

    # pad by edge
    padded = np.pad(kp_1TVC, ((0, 0), (pad, pad), (0, 0), (0, 0)), mode="edge")

    for t in range(T):
        win = padded[:, t:t + k, :, :]  # (1,k,V,C)
        if mode == "median":
            out[:, t, :, :] = np.nanmedian(win, axis=1)
        else:
            out[:, t, :, :] = np.nanmean(win, axis=1)
    return out


def stitch_by_prefix(annos, prefix, gap_policy="skip", boundary_pad=0):
    segs = [a for a in annos if a.get("frame_dir", "").startswith(prefix)]
    if len(segs) == 0:
        raise ValueError(f"No segments matched group_prefix={prefix}")

    segs = sorted(segs, key=lambda a: int(a.get("start_frame", 0)))

    print(f"Matched {len(segs)} segments for group_prefix={prefix}")
    for a in segs:
        print(f"   {a.get('frame_dir')} start={int(a.get('start_frame',0)):4d} end={int(a.get('end_frame',0)):4d} label={a.get('label')}")

    kp0 = segs[0]["keypoint"]
    _, _, V, C = kp0.shape

    # timeline length
    T_total = int(max(int(a.get("end_frame")) for a in segs))
    if boundary_pad > 0:
        T_total = T_total + 2 * boundary_pad

    stitched = np.full((1, T_total, V, C), np.nan, dtype=np.float32)
    segname = np.array([""] * T_total, dtype=object)

    for a in segs:
        kp = a["keypoint"].astype(np.float32)  # (1, Tseg, V, C)
        start = int(a.get("start_frame"))
        end = int(a.get("end_frame"))

        if boundary_pad > 0:
            start += boundary_pad
            end += boundary_pad

        start = max(0, start)
        end = min(T_total, end)
        Tseg = min(kp.shape[1], end - start)

        stitched[:, start:start + Tseg, :, :] = kp[:, :Tseg, :, :]
        segname[start:start + Tseg] = a.get("frame_dir", prefix)

    if gap_policy == "pad":
        last = None
        for t in range(T_total):
            frame = stitched[0, t]
            if np.isnan(frame).all():
                if last is not None:
                    stitched[0, t] = last
                    segname[t] = segname[t] or "PAD"
            else:
                last = stitched[0, t].copy()

        # backfill beginning if needed
        if np.isnan(stitched[0, 0]).all():
            first_valid = None
            for t in range(T_total):
                if not np.isnan(stitched[0, t]).all():
                    first_valid = stitched[0, t].copy()
                    break
            if first_valid is not None:
                for t in range(T_total):
                    if np.isnan(stitched[0, t]).all():
                        stitched[0, t] = first_valid
                        segname[t] = segname[t] or "PAD"
                    else:
                        break

        valid_mask = ~np.isnan(stitched[0, :, 0, 0])
        stitched = stitched[:, valid_mask, :, :]
        segname = segname[valid_mask]

    else:  # skip
        valid_mask = ~np.isnan(stitched[0, :, 0, 0])
        stitched = stitched[:, valid_mask, :, :]
        segname = segname[valid_mask]

    return stitched, segname, segs


def stitch_youtube_range(annos, youtube_start, youtube_end,
                         seconds_per_segment=None, source_fps=None):
    """
    Stitch segments from YouTube_XXX where XXX in [youtube_start, youtube_end],
    sorted first by video id, then by start_frame.

    If seconds_per_segment is not None, keep only the first
    round(seconds_per_segment * source_fps) frames from each segment.
    """
    matched = []

    for a in annos:
        frame_dir = str(a.get("frame_dir", ""))
        if not frame_dir.startswith("YouTube_"):
            continue

        # parse id from e.g. YouTube_001_posture3
        parts = frame_dir.split("_")
        if len(parts) < 2:
            continue

        try:
            vid_id = int(parts[1])
        except Exception:
            continue

        if youtube_start <= vid_id <= youtube_end:
            matched.append((vid_id, a))

    if len(matched) == 0:
        raise ValueError(
            f"No segments found for YouTube_{youtube_start:03d} to YouTube_{youtube_end:03d}"
        )

    # sort by YouTube id, then start_frame
    matched = sorted(matched, key=lambda x: (x[0], int(x[1].get("start_frame", 0))))

    print(f"Matched {len(matched)} segments for YouTube_{youtube_start:03d} to YouTube_{youtube_end:03d}")
    for vid_id, a in matched:
        print(
            f"   {a.get('frame_dir')} "
            f"start={int(a.get('start_frame', 0)):4d} "
            f"end={int(a.get('end_frame', 0)):4d} "
            f"label={a.get('label')}"
        )

    if seconds_per_segment is not None:
        if source_fps is None:
            raise ValueError("source_fps must be provided when seconds_per_segment is used")
        keep_frames = int(round(seconds_per_segment * source_fps))
        print(f"Keeping first {keep_frames} frames per segment "
              f"({seconds_per_segment}s at {source_fps} fps)")
    else:
        keep_frames = None

    stitched_list = []
    segname_list = []

    for vid_id, a in matched:
        kp = a["keypoint"].astype(np.float32)  # (1, T, V, C)

        # keep only first N frames if requested
        if keep_frames is not None:
            kp = kp[:, :min(kp.shape[1], keep_frames), :, :]

        Tseg = kp.shape[1]
        if Tseg == 0:
            continue

        stitched_list.append(kp)
        segname_list.extend([a.get("frame_dir", f"YouTube_{vid_id:03d}")] * Tseg)

    if len(stitched_list) == 0:
        raise ValueError("All matched segments became empty after truncation")

    stitched = np.concatenate(stitched_list, axis=1)  # (1, T_total, V, C)
    segname = np.array(segname_list, dtype=object)

    return stitched, segname

# --------------------------
# Main
# --------------------------
def main():
    args = parse_args()

    repo_root = infer_repo_root(args)
    pyskl_root = os.path.join(repo_root, "pyskl")
    if pyskl_root not in sys.path:
        sys.path.insert(0, pyskl_root)

    if args.config is None:
        args.config = os.path.join(pyskl_root, "configs/dgstgcn/infact_primitive_2dkp/j.py")
    if args.checkpoint is None:
        args.checkpoint = os.path.join(pyskl_root, "work_dirs/dgstgcn/infact_primitive_2dkp/j/latest.pth")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)

    print(f"repo_root      : {repo_root}")
    print(f"pyskl_root     : {pyskl_root}")
    print(f"config         : {args.config}")
    print(f"checkpoint     : {args.checkpoint}")
    print(f"pkl            : {os.path.abspath(args.pkl)}")
    print(f"out            : {out_path}")

    for p in [args.config, args.checkpoint, args.pkl]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    print("CUDA available :", torch.cuda.is_available())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", None))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device   :", device)

    from pyskl.models import build_model
    from pyskl.datasets.pipelines import Compose

    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model).to(device)
    _ = load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    # expected_C from BatchNorm features
    V_cfg = cfg.model["backbone"]["graph_cfg"].get("num_node", 17)
    bn_features = int(model.backbone.data_bn.num_features)
    expected_C = bn_features // int(V_cfg)
    print(f"Model expects: V={V_cfg}, data_bn_features={bn_features} => expected_C={expected_C}")

    # minimal pipeline consistent with training feature generation
    pipe_cfg = [
        dict(type="PreNormalize3D", align_spine=False),
        dict(type="GenSkeFeat", feats=["j"], dataset="coco"),
        dict(type="PoseDecode"),
        dict(type="FormatGCNInput", num_person=1),
        dict(type="ToTensor", keys=["keypoint"]),
    ]
    demo_pipeline = Compose(pipe_cfg)

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    annos = data["annotations"]
    segname_t = None

    # choose keypoints
    if args.youtube_start is not None and args.youtube_end is not None:
        kp_full, segname_t = stitch_youtube_range(
            annos,
            youtube_start=args.youtube_start,
            youtube_end=args.youtube_end,
            seconds_per_segment=args.seconds_per_segment,
            source_fps=args.source_fps
        )
        frame_dir_show = f"YouTube_{args.youtube_start:03d}_to_YouTube_{args.youtube_end:03d}"
        ann_for_meta = next(a for a in annos if str(a.get("frame_dir", "")).startswith("YouTube_"))

    elif args.group_prefix is not None:
        kp_full, segname_t, _segs = stitch_by_prefix(
            annos,
            args.group_prefix,
            gap_policy=args.gap_policy,
            boundary_pad=args.boundary_pad
        )
        frame_dir_show = args.group_prefix
        ann_for_meta = next(a for a in annos if a.get("frame_dir", "").startswith(args.group_prefix))

    else:
        if args.frame_dir is not None:
            frame_dir = args.frame_dir
        else:
            split_list = data["split"][args.split]
            frame_dir = split_list[args.index]

        idx = next(i for i, a in enumerate(annos) if a.get("frame_dir") == frame_dir)
        ann_for_meta = annos[idx]
        kp_full = ann_for_meta["keypoint"].astype(np.float32)
        frame_dir_show = frame_dir

    # enforce expected_C once globally (safe)
    if kp_full.shape[-1] != expected_C:
        kp_full = kp_full[..., :expected_C].astype(np.float32)

    # optional keypoint smoothing (recommended for D02)
    if args.kp_smooth and args.kp_smooth > 1:
        kp_full = temporal_smooth_kp(kp_full, k=args.kp_smooth, mode=args.kp_smooth_mode)

    T = kp_full.shape[1]
    V = kp_full.shape[2]
    C = kp_full.shape[3]
    print("Picked:", frame_dir_show)
    print("keypoint shape:", kp_full.shape)
    print("T:", T, "V:", V, "C:", C)

    @torch.no_grad()
    def predict_window(window_1TVC):
        # window_1TVC is already expected_C due to global enforcement, but keep safe:
        if window_1TVC.shape[-1] != expected_C:
            window_1TVC = window_1TVC[..., :expected_C]

        sample = {
            "keypoint": window_1TVC.astype(np.float32),
            "total_frames": window_1TVC.shape[1],
            "label": 0,
            "img_shape": ann_for_meta.get("img_shape", (1080, 1920)),
            "original_shape": ann_for_meta.get("original_shape", (1080, 1920)),
        }
        sample = demo_pipeline(sample)
        keypoint = sample["keypoint"]  # tensor

        # normalize to (B, num_clips, P, T, V, C)
        if keypoint.dim() == 4:            # (P, T, V, C)
            keypoint = keypoint.unsqueeze(0).unsqueeze(0)
        elif keypoint.dim() == 5:          # (num_clips, P, T, V, C)
            keypoint = keypoint.unsqueeze(0)
        else:
            raise RuntimeError(f"Unexpected keypoint shape: {tuple(keypoint.shape)}")

        keypoint = keypoint.to(device)
        out = model(return_loss=False, keypoint=keypoint)

        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, np.ndarray):
            logits = torch.from_numpy(out).float()
        else:
            logits = out.detach().float().cpu()

        logits = logits.reshape(-1)
        probs = torch.softmax(logits, dim=0).numpy()
        topk = np.argsort(-probs)[:max(1, args.show_topk)]
        pred = int(topk[0])
        conf = float(probs[pred])
        margin = float(probs[topk[0]] - probs[topk[1]]) if len(topk) >= 2 else float("nan")
        return pred, conf, margin, probs, topk

    # optional prediction vote for display
    vote_buf = []  # store recent preds

    frames = []
    max_t = min(T, args.max_frames)
    render_i = 0

    for t in range(0, max_t, args.step):
        w = window_at_t(kp_full, t, args.clip_len)
        pred, conf, margin, probs, topk = predict_window(w)

        # update vote buffer
        if args.pred_vote and args.pred_vote > 1:
            vote_buf.append(pred)
            if len(vote_buf) > args.pred_vote:
                vote_buf.pop(0)
            # majority vote for display
            vals, cnts = np.unique(np.array(vote_buf, dtype=int), return_counts=True)
            pred_show = int(vals[np.argmax(cnts)])
        else:
            pred_show = pred

        # uncertain logic (based on raw pred margin)
        uncertain = (np.isfinite(margin) and margin < args.uncertain_margin)

        # choose color (gray if uncertain, else by displayed pred)
        if uncertain:
            color = COLOR_GRAY
        else:
            color = LABEL_COLORS.get(pred_show, (0.2, 0.2, 0.2))

        fig, ax = plt.subplots(figsize=(7, 5))
        draw_skeleton_2d(ax, kp_full[0, t], color=color)

        # build title
        if segname_t is not None:
            segtxt = str(segname_t[t])
            prefix = f"{frame_dir_show} | seg={segtxt} | t={t:04d}"
        else:
            prefix = f"{frame_dir_show} | t={t:04d}"

        # topk string
        topk_str = ", ".join([f"{LABEL_NAMES[i]}:{probs[i]:.3f}" if i < len(LABEL_NAMES) else f"{i}:{probs[i]:.3f}" for i in topk])
        if np.isfinite(margin):
            tail = f" | top{len(topk)} [{topk_str}] | margin={margin:.3f}"
        else:
            tail = f" | top{len(topk)} [{topk_str}]"

        if args.pred_vote and args.pred_vote > 1:
            pred_vote_name = LABEL_NAMES[pred_show] if pred_show < len(LABEL_NAMES) else str(pred_show)
            tail += f" | vote({args.pred_vote})={pred_vote_name}"

        if uncertain:
            tail += " | UNCERTAIN"

        title = prefix + tail

        # title top-center, avoid covering skeleton
        fig.subplots_adjust(top=0.84)
        fig.suptitle(title, y=0.98, fontsize=10)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)

        render_i += 1
        if args.debug_every and args.debug_every > 0 and (render_i % args.debug_every == 0 or t == 0):
            top1 = topk[0]
            top2 = topk[1] if len(topk) > 1 else None
            msg = f"Rendered t={t}/{max_t} pred={LABEL_NAMES[top1]} conf={conf:.4f}"
            if top2 is not None:
                msg += f" second={LABEL_NAMES[top2]} {probs[top2]:.4f} margin={margin:.4f}"
            if args.pred_vote and args.pred_vote > 1:
                msg += f" pred_show={LABEL_NAMES[pred_show]}"
            if uncertain:
                msg += " UNCERTAIN"
            print(msg)

    imageio.mimsave(out_path, frames, fps=args.fps)
    print("Saved GIF:", out_path)


if __name__ == "__main__":
    main()
