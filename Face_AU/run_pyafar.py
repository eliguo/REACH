import argparse
import os
import sys
import pandas as pd

from adult_afar import adult_afar, set_models_dir as set_adult_models_dir
from infant_afar import infant_afar, set_models_dir as set_infant_models_dir

INFANT_OCC = ["au_1", "au_2", "au_3", "au_4", "au_6", "au_9", "au_12", "au_20", "au_28"]
ADULT_OCC = ["au_1", "au_2", "au_4", "au_6", "au_7", "au_10", "au_12", "au_14", "au_15", "au_17", "au_23", "au_24"]
ADULT_INT = ["au_6", "au_10", "au_12", "au_14", "au_17"]


def parse_list(value, default_list):
    if value is None or value.strip() == "":
        return default_list
    parts = [v.strip() for v in value.split(",")]
    return [p for p in parts if p]


def build_arg_parser():
    p = argparse.ArgumentParser(description="Run PyAFAR programmatically (no GUI).")
    p.add_argument("--mode", choices=["adult", "infant"], required=True)
    p.add_argument("--input", required=True, help="Path to input video file")
    p.add_argument("--output", required=True, help="Path to output CSV file")
    p.add_argument("--aus", default=None, help="Comma-separated AU list (e.g., au_1,au_2). Defaults to mode-specific list.")
    p.add_argument("--au-int", default=None, help="Comma-separated AU intensity list (adult only). Defaults to adult intensity list.")
    p.add_argument("--gpu", type=int, default=0, help="1 to enable GPU, 0 to disable")
    p.add_argument("--max-frames", type=int, default=1000, help="Preprocessing batch size")
    p.add_argument("--batch-size", type=int, default=100, help="Prediction batch size")
    p.add_argument("--pid", action="store_true", help="Enable person tracking (adult only)")
    p.add_argument("--fill-value", type=float, default=0, help="Fill value for missing-face frames")
    p.add_argument("--models-dir", default=None, help="Path to PyAFAR models directory")
    return p


def main():
    args = build_arg_parser().parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.models_dir:
        set_adult_models_dir(args.models_dir)
        set_infant_models_dir(args.models_dir)
    else:
        env_models = os.environ.get("PYAFAR_MODELS_DIR")
        if env_models:
            set_adult_models_dir(env_models)
            set_infant_models_dir(env_models)

    if args.mode == "infant":
        aus = parse_list(args.aus, INFANT_OCC)
        data = infant_afar(args.input, aus, bool(args.gpu), args.max_frames, fill_value=args.fill_value)
    else:
        aus = parse_list(args.aus, ADULT_OCC)
        au_int = parse_list(args.au_int, ADULT_INT)
        data = adult_afar(
            args.input,
            aus,
            bool(args.gpu),
            args.max_frames,
            au_int,
            args.batch_size,
            args.pid,
            fill_value=args.fill_value,
        )

    df = pd.DataFrame.from_dict(data)
    df.to_csv(args.output, index=False)
    print(f"Saved CSV to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
python run_pyafar.py \
  --mode infant \
  --input mediapipe/still_in/Still face experiment - Lise-Lotte Austad (720p, h264).mp4 \
  --output mediapipe/still_out/SFE.csv
"""
