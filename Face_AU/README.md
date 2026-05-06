# PyAFAR Facial AU Extraction And Visualization

This folder provides command-line wrappers and notebooks for extracting facial
Action Units (AUs) from videos with PyAFAR-style adult and infant models, then
visualizing the resulting AU and landmark CSV files.

Typical workflow:

1. Install PyAFAR and obtain the model files by following the official PyAFAR
   instructions at <https://pyafar.org/>.
2. Activate the conda environment where PyAFAR and its dependencies are
   installed.
3. Run AU extraction with `run_pyafar.py` or `run_pyafar_multi.py`.
4. Inspect the output CSV with the visualization notebooks.

## Folder Contents

| Path | Purpose |
| --- | --- |
| `run_pyafar.py` | Single-face command-line runner. |
| `run_pyafar_multi.py` | Multi-face command-line runner. Also useful for one-face output with `--max-num-faces 1`. |
| `infant_afar.py` | Infant single-face processing: MediaPipe landmarks, dlib alignment, HOG features, LightGBM AU occurrence models. |
| `infant_afar_multi.py` | Infant multi-face processing. Produces one row per detected face per frame. |
| `adult_afar.py` | Adult single-face processing: MediaPipe landmarks, dlib alignment, Keras AU occurrence/intensity models. |
| `adult_afar_multi.py` | Adult multi-face processing. Produces one row per detected face per frame. |
| `PyAFAR_GUI.py` | Older GUI/batch wrapper and model download/check logic. The CLI scripts are the recommended entry points here. |
| `landmark_viz_video.ipynb` | Single-face visualization notebook. |
| `landmark_viz_video_multi.ipynb` | Multi-face visualization notebook. |
| `new_evaluations.ipynb` | Example display-only AU summary/statistics notebook. |
| `demo_vids/` | Example/demo videos, generated CSVs, and visual outputs. |
| `still_in/` | Example input videos. |
| `still_out/` | Example AU/landmark CSV outputs. |
| `vis_still_multi/` | Example multi-face visualization outputs. |

## Environment Setup

Create or activate a conda environment that contains PyAFAR and the required
dependencies. Replace `<ENV_NAME>` with your environment name.

```bash
conda activate <ENV_NAME>
```

The scripts depend on packages such as:

- `opencv-python` / `cv2`
- `mediapipe`
- `dlib`
- `tensorflow`
- `lightgbm`
- `numpy`
- `pandas`
- `scipy`
- `scikit-image`
- `matplotlib`
- `tqdm`

The visualization notebooks also require `ffmpeg` on the system path because
Matplotlib uses `FFMpegWriter` to render MP4 files.

## PyAFAR Models

The AU extraction scripts require the PyAFAR model files. Follow the official
PyAFAR installation/model instructions at:

```text
https://pyafar.org/
```

After installing PyAFAR into a conda environment, the models are commonly under
the environment's `site-packages/PyAFAR_GUI/models` directory.

If you use Miniforge installed under your home directory, the path often looks
like this:

```bash
$HOME/miniforge3/envs/<ENV_NAME>/lib/python3.9/site-packages/PyAFAR_GUI/models
```

A more portable way is to use `$CONDA_PREFIX` after activating the environment:

```bash
conda activate <ENV_NAME>
export PYAFAR_MODELS_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/PyAFAR_GUI/models"
```

You can then either pass the model directory explicitly:

```bash
--models-dir "$PYAFAR_MODELS_DIR"
```

or rely on the `PYAFAR_MODELS_DIR` environment variable.

The model directory should contain files such as:

- `shape_predictor_5_face_landmarks.dat`
- infant LightGBM AU models under `Infant/`
- adult occurrence models under `adult/occ/`
- adult intensity models under `adult/int/`

## AU Extraction

Run all commands from this folder, or replace `<PROJECT_DIR>` with the absolute
path to this folder.

```bash
cd <PROJECT_DIR>
conda activate <ENV_NAME>
export PYAFAR_MODELS_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/PyAFAR_GUI/models"
```

### Single-Face Runner

Use `run_pyafar.py` when you want one row per video frame and only the first
detected face is needed.

```bash
python run_pyafar.py \
  --mode infant \
  --input "<INPUT_VIDEO>" \
  --output "<OUTPUT_CSV>" \
  --models-dir "$PYAFAR_MODELS_DIR" \
  --fill-value 0
```

### Multi-Face Runner

Use `run_pyafar_multi.py` when the video may contain multiple faces, or when you
want a `Face_Index` column in the output CSV.

```bash
python run_pyafar_multi.py \
  --mode infant \
  --input "<INPUT_VIDEO>" \
  --output "<OUTPUT_CSV>" \
  --models-dir "$PYAFAR_MODELS_DIR" \
  --max-num-faces <MAX_NUM_FACES> \
  --fill-value 0
```

For a one-face analysis with the multi-face output format, use:

```bash
--max-num-faces 1
```

## Examples

### Infant, One Face

```bash
cd <PROJECT_DIR>
conda activate <ENV_NAME>
export PYAFAR_MODELS_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/PyAFAR_GUI/models"

python run_pyafar_multi.py \
  --mode infant \
  --input "demo_vids/<VIDEO_NAME>.mp4" \
  --output "demo_vids/<OUTPUT_NAME>.csv" \
  --models-dir "$PYAFAR_MODELS_DIR" \
  --max-num-faces 1 \
  --fill-value 0
```

Default infant AU occurrence outputs:

```text
Occ_au_1, Occ_au_2, Occ_au_3, Occ_au_4, Occ_au_6,
Occ_au_9, Occ_au_12, Occ_au_20, Occ_au_28
```

### Infant, Multiple Faces

```bash
cd <PROJECT_DIR>
conda activate <ENV_NAME>
export PYAFAR_MODELS_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/PyAFAR_GUI/models"

python run_pyafar_multi.py \
  --mode infant \
  --input "<INPUT_VIDEO>" \
  --output "<OUTPUT_CSV>" \
  --models-dir "$PYAFAR_MODELS_DIR" \
  --max-num-faces 5 \
  --fill-value 0
```

The output will include `Face_Index`, where each detected face in a frame gets
its own row.

### Adult

```bash
cd <PROJECT_DIR>
conda activate <ENV_NAME>
export PYAFAR_MODELS_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/PyAFAR_GUI/models"

python run_pyafar_multi.py \
  --mode adult \
  --input "<INPUT_VIDEO>" \
  --output "<OUTPUT_CSV>" \
  --models-dir "$PYAFAR_MODELS_DIR" \
  --max-num-faces 1 \
  --fill-value 0
```

Default adult occurrence outputs:

```text
Occ_au_1, Occ_au_2, Occ_au_4, Occ_au_6, Occ_au_7, Occ_au_10,
Occ_au_12, Occ_au_14, Occ_au_15, Occ_au_17, Occ_au_23, Occ_au_24
```

Default adult intensity outputs:

```text
Int_au_6, Int_au_10, Int_au_12, Int_au_14, Int_au_17
```

## Batch Or Cluster Job Template

If running on a compute cluster, place the same commands inside your scheduler's
job script. The scheduler directives vary by institution, so treat the header
below as a placeholder.

```bash
#!/bin/bash
# <SCHEDULER_DIRECTIVES_GO_HERE>

set -euo pipefail

cd <PROJECT_DIR>
source <PATH_TO_CONDA_INITIALIZATION_SCRIPT>
conda activate <ENV_NAME>

export PYAFAR_MODELS_DIR="$CONDA_PREFIX/lib/python3.9/site-packages/PyAFAR_GUI/models"

python run_pyafar_multi.py \
  --mode <infant_or_adult> \
  --input "<INPUT_VIDEO>" \
  --output "<OUTPUT_CSV>" \
  --models-dir "$PYAFAR_MODELS_DIR" \
  --max-num-faces <MAX_NUM_FACES> \
  --fill-value 0
```

Examples of scheduler-specific submission commands might look like
`sbatch <JOB_SCRIPT>` or `qsub <JOB_SCRIPT>`, depending on the system. Use your
cluster's documentation.

## CLI Options

| Option | Meaning |
| --- | --- |
| `--mode infant` / `--mode adult` | Selects the infant or adult model family. |
| `--input` | Input video path. |
| `--output` | Output CSV path. Parent folders are created automatically. |
| `--models-dir` | Path to the PyAFAR model directory. |
| `--aus` | Optional comma-separated occurrence AU list, for example `au_1,au_2,au_12`. |
| `--au-int` | Optional comma-separated adult intensity AU list. Adult mode only. |
| `--gpu 1` | Enables GPU by setting CUDA device `0`. Default is CPU mode, `--gpu 0`. |
| `--max-frames` | Preprocessing batch size. Default: `1000`. |
| `--batch-size` | Adult Keras prediction batch size. Default: `100`. |
| `--pid` | Adult person tracking option. Leave off unless the FaceNet/person-tracking model path has been verified. |
| `--fill-value` | Value written for missing-face frames. Default: `0`. |
| `--max-num-faces` | Multi-face runner only. Maximum number of faces MediaPipe should detect per frame. |

## Output CSV Format

Single-face output has one row per frame. Multi-face output has one row per
detected face per frame.

Common columns:

| Column | Meaning |
| --- | --- |
| `Frame` | Video frame number. |
| `Face_Index` | Multi-face output only. Detected face index within the frame. Missing-face rows use `-1`. |
| `Pitch`, `Yaw`, `Roll` | Head-pose estimates. |
| `Eye Aspect Ratio` | Eye-opening metric computed from landmark distances. |
| `Mouth Aspect Ratio` | Mouth-opening metric computed from landmark distances. |
| `Occ_au_*` | AU occurrence prediction score. |
| `Int_au_*` | Adult AU intensity output, when requested. |
| `x_0`, `y_0`, `z_0`, ..., `x_467`, `y_467`, `z_467` | MediaPipe face landmark coordinates. |

Frames where no face is detected are filled with `--fill-value`.

## Visualization Notebooks

Use the same conda environment for notebooks:

```bash
cd <PROJECT_DIR>
conda activate <ENV_NAME>
jupyter notebook
```

### `landmark_viz_video.ipynb`

Use this notebook for single-face CSV outputs.

Typical steps:

1. Set `CSV_PATH` to the AU/landmark CSV.
2. Set `OUTDIR` to the visualization output folder.
3. Set the relevant `VIDEO_PATH` cells to the source video.
4. Run the cells that load the CSV, detect landmark columns, and build the mesh.
5. Generate landmark-only, overlay, time-series, or combined visualizations.

Example path pattern:

```python
CSV_PATH = "<PATH_TO_SINGLE_FACE_CSV>"
OUTDIR = "<PATH_TO_VISUALIZATION_OUTPUT_FOLDER>"
VIDEO_PATH = "<PATH_TO_SOURCE_VIDEO>"
```

### `landmark_viz_video_multi.ipynb`

Use this notebook for multi-face CSV outputs that include `Face_Index`.

Typical steps:

1. Set `CSV_PATH` to the multi-face AU/landmark CSV.
2. Set `OUTDIR` to the visualization output folder.
3. Set `VIDEO_PATH` to the matching source video.
4. Run the cells that group rows by `Frame` and `Face_Index`.
5. Generate multi-face landmark-only, overlay, time-series, or combined
   visualizations.

Example path pattern:

```python
CSV_PATH = "<PATH_TO_MULTI_FACE_CSV>"
OUTDIR = "<PATH_TO_VISUALIZATION_OUTPUT_FOLDER>"
VIDEO_PATH = "<PATH_TO_SOURCE_VIDEO>"
```

### `new_evaluations.ipynb`

This notebook is a display-only example for summarizing selected AU columns
within a specified frame/time window. It shows statistics and plots inside the
notebook and does not write new output files.

## Optional Video Conversion

There is no dedicated MOV-to-MP4 conversion script in this folder. Use `ffmpeg`
directly if a video needs conversion or frame-rate adjustment:

```bash
ffmpeg -i "<INPUT_MOV>" -r <TARGET_FPS> -c:v libx264 -pix_fmt yuv420p -c:a aac "<OUTPUT_MP4>"
```

## Practical Notes

- Make sure the source video and output CSV use the same frame timeline before
  creating overlay videos.
- Use `--max-num-faces 1` for single-subject videos when using
  `run_pyafar_multi.py`.
- Multi-face CSVs are in long format: multiple rows may share the same `Frame`
  but have different `Face_Index` values.
- The notebooks assume landmark coordinates are normalized relative to the full
  video frame.
- Generated videos and CSVs can be large; keep only the outputs needed for your
  analysis.

## Quick Checklist

Before extraction:

1. Follow PyAFAR setup/model instructions at <https://pyafar.org/>.
2. Activate the conda environment: `conda activate <ENV_NAME>`.
3. Confirm the model directory exists:
   `$CONDA_PREFIX/lib/python3.9/site-packages/PyAFAR_GUI/models`.
4. Choose `--mode infant` or `--mode adult`.
5. Choose `run_pyafar.py` for single-face or `run_pyafar_multi.py` for
   multi-face.
6. Set `<INPUT_VIDEO>` and `<OUTPUT_CSV>`.

Before visualization:

1. Open the matching notebook.
2. Set `CSV_PATH`.
3. Set `VIDEO_PATH`.
4. Set `OUTDIR` if the notebook cell writes visualization outputs.
5. Run cells from top to bottom unless you know exactly which variables have
   already been created.
