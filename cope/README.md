# cope

Inventory and audit tooling for the COPE longitudinal video dataset (parent-recorded infant–caregiver interactions at 6, 12, and 42 months).

## What it does

`cope.py` walks the COPE video tree, parses subject IDs / tasks / phases out of paths and filenames, pulls per-video durations via `ffprobe`, and writes a set of summary CSVs to `outputs/`:

- `top_level_summary.csv` — per-timepoint item / size totals
- `filetype_summary.csv` — file type breakdown
- `video_summary.csv` — per-timepoint video count and duration stats
- `task_summary.csv` — videos per task
- `subject_task_summary.csv` — per-subject × phase task coverage and what's missing
- `unparsed_files.csv` — files that didn't match any task pattern (sanity check)
- `annotation_files.csv` — experimenter session notes (candidate label source)
- `annotation_coverage.csv` — note coverage by timepoint × phase
- `dataset_inventory_full.csv` — full per-file table

## Usage

Edit `ROOT` at the top of `cope.py` to point at your COPE mount, then:

```bash
python cope.py
```

Requires `ffmpeg` on `$PATH` and `pandas`, `numpy`, `tqdm` installed.

## Notes

The COPE folder structure is messy in places — typo'd task names, mixed flat / nested layouts across timepoints, date-stamped filenames, etc. The parser handles known variants explicitly (see `TASK_PATTERNS` in `cope.py`); anything that still slips through lands in `unparsed_files.csv` for manual review.
