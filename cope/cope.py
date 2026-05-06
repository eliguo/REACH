"""
COPE video dataset inventory.

Walks the COPE video tree, parses subject/task/phase from paths and filenames,
pulls durations via ffprobe, and dumps a bunch of summary CSVs to ./outputs/.

Outputs:
    top_level_summary.csv
    filetype_summary.csv
    video_summary.csv
    task_summary.csv
    subject_task_summary.csv
    unparsed_files.csv
    annotation_files.csv      -- experimenter notes, candidate training labels
    annotation_coverage.csv   -- per-timepoint label coverage

Run: python cope.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT   = Path("/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential")
OUTDIR = Path("./outputs")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".m4v", ".3gp"}
DOC_EXTS   = {".pdf", ".docx", ".pages", ".doc"}

EXPECTED_6MO_TASKS = {
    "arm_restraint", "freeplay", "nail_clipping", "still_face", "visual_attention",
}
EXPECTED_12MO_TASKS = {
    "arm_restraint", "freeplay", "nail_clipping", "still_face", "maap",
}
EXPECTED_42MO_TASKS = {
    "quills", "picnic", "gift", "puzzles", "active_webcam", "passive_webcam",
}

# order matters: typo / specific variants must come before canonical spellings
TASK_PATTERNS: list[tuple[str, str]] = [
    # typos
    (r"freplay",                    "freeplay"),
    (r"clippig",                    "nail_clipping"),
    (r"nail[_ ]?cutting",           "nail_clipping"),
    (r"arm[_ ]?retraint",           "arm_restraint"),
    (r"mobile[_ ]?still[_ ]?face",  "still_face"),
    (r"visual[_ ]?attnetion",       "visual_attention"),
    (r"mapp\b",                     "maap"),
    # 42mo screen / pip recordings of webcam tasks
    (r"active[_ ]?screen",          "active_webcam"),
    (r"passive[_ ]?screen",         "passive_webcam"),
    (r"active[_ ]?pip",              "active_webcam"),
    (r"passive[_ ]?pip",             "passive_webcam"),
    # jumble variants
    (r"jumble[_ ]?scscreen",        "jumble"),
    (r"jumble[_ ]?pip",             "jumble"),
    (r"jumble[_ ]?screen",          "jumble"),
    (r"jumble[_ ]?webcam",          "jumble"),
    (r"jumble",                     "jumble"),
    # canonical
    (r"arm[_ ]?restraint",          "arm_restraint"),
    (r"free[_ ]?play",              "freeplay"),
    (r"nail[_ ]?clipping",          "nail_clipping"),
    (r"still[_ ]?face",             "still_face"),
    (r"visual[_ ]?attention",       "visual_attention"),
    (r"maap",                       "maap"),
    (r"depth",                      "depth"),
    (r"calibration",                "calibration"),
    (r"puzzle|puzzles",             "puzzles"),
    (r"gift",                       "gift"),
    (r"picnic",                     "picnic"),
    (r"science",                    "science"),
    (r"bonus",                      "bonus"),
    (r"zoom",                       "zoom"),
    (r"quils|quills",               "quills"),
    (r"active[_ ]?webcam",          "active_webcam"),
    (r"passive[_ ]?webcam",         "passive_webcam"),
    (r"episode|ep[_ ]?\d+",         "episode"),
    (r"full[_ ]?video",             "full_video"),
    # 42mo misc
    (r"fnirs",                      "fnirs"),
    (r"musicbox",                   "musicbox"),
    (r"epilog",                     "epilog"),
    (r"cleanup",                    "cleanup"),
    (r"verbal[_ ]?exercise",        "verbal_exercise"),
    # notes goes last so it doesn't shadow real task names
    (r"notes",                      "notes"),
]


_YEAR_PAT    = re.compile(r"^20\d{2}$")
_SUBJECT_PAT = re.compile(r"^[A-Za-z]?\d+(?:-\d+)?$")


def _is_subject_id(s: str) -> bool:
    # subject IDs look like 100, M21618, 12-3 etc. but NOT calendar years
    return bool(_SUBJECT_PAT.match(s)) and not bool(_YEAR_PAT.match(s))


def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("-", "_")
    s = re.sub(r"[() ]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s


def get_top_level(rel: Path) -> str | None:
    return rel.parts[0] if rel.parts else None


def parse_subject_id(rel: Path) -> str | None:
    parts = rel.parts
    if len(parts) == 1:
        return None

    top = parts[0]

    # month-based folders: 6_Months/100/E1/...  or  42_Months/Task Videos & Notes/M21618/...
    # walk components after top-level, first valid subject ID wins
    if re.fullmatch(r"\d+_Months", top):
        for part in parts[1:]:
            if _is_subject_id(part):
                return part.upper()

    # flat aggregated folders, e.g.
    #   Freeplay_6mo/M21052_6_Freeplay.mov
    #   MAAP_Cropped_Videos/COPE_100_12_MAAP.mp4
    m = re.match(r"^(?:COPE_)?([A-Za-z]?\d+(?:-\d+)?)", rel.name)
    if m:
        candidate = m.group(1)
        if _is_subject_id(candidate):
            return candidate.upper()

    return None


def parse_age_month(rel: Path) -> int | None:
    top = get_top_level(rel)
    if top:
        m = re.match(r"^(\d+)_Months$", top)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)mo", top.lower())
        if m:
            return int(m.group(1))
    # fallback: _6_, _12_, _42_ in filename
    m = re.search(r"_(\d+)_", rel.name)
    if m:
        val = int(m.group(1))
        if val in (6, 12, 42):
            return val
    return None


def parse_phase(rel: Path) -> str | None:
    for p in rel.parts:
        if p in {"E1", "E2", "E3"}:
            return p
    return None


def parse_task(name: str) -> str | None:
    s = normalize_text(name)
    for pat, task in TASK_PATTERNS:
        if re.search(pat, s):
            return task
    return None


def parse_task_variant(name: str) -> str | None:
    """Fine-grained variant within a task, e.g. freeplay_no_toys, active_webcam_pip."""
    s = normalize_text(name)

    if re.search(r"freeplay|freplay|free_play", s):
        if "no_toys" in s:
            return "freeplay_no_toys"
        if "with_toys" in s or re.search(r"(^|_)toys($|_)", s):
            return "freeplay_with_toys"
        return "freeplay"

    if re.search(r"arm[_ ]?restraint|arm[_ ]?retraint", s): return "arm_restraint"
    if re.search(r"nail[_ ]?clipping|clippig|nail[_ ]?cutting", s): return "nail_clipping"
    if re.search(r"still[_ ]?face|mobile[_ ]?still[_ ]?face", s): return "still_face"
    if re.search(r"visual[_ ]?attention|attnetion", s): return "visual_attention"
    if re.search(r"maap|mapp\b", s): return "maap"

    # webcam sub-types
    if re.search(r"active[_ ]?screen",  s): return "active_webcam_screen"
    if re.search(r"passive[_ ]?screen", s): return "passive_webcam_screen"
    if re.search(r"active[_ ]?pip",     s): return "active_webcam_pip"
    if re.search(r"passive[_ ]?pip",    s): return "passive_webcam_pip"
    if re.search(r"active[_ ]?webcam",  s): return "active_webcam"
    if re.search(r"passive[_ ]?webcam", s): return "passive_webcam"

    if "jumble" in s:
        if re.search(r"scscreen|screen", s): return "jumble_screen"
        if "pip"    in s:                    return "jumble_pip"
        if "webcam" in s:                    return "jumble_webcam"
        return "jumble"

    if "fnirs"    in s: return "fnirs"
    if "musicbox" in s: return "musicbox"
    if "epilog"   in s: return "epilog"
    if "cleanup"  in s: return "cleanup"
    if re.search(r"verbal[_ ]?exercise", s): return "verbal_exercise"
    if "depth"       in s: return "depth"
    if "calibration" in s: return "calibration"
    if "puzzle"      in s: return "puzzles"
    if "gift"        in s: return "gift"
    if "picnic"      in s: return "picnic"
    if "science"     in s: return "science"
    if "bonus"       in s: return "bonus"
    if "zoom"        in s: return "zoom"
    if re.search(r"quils|quills", s): return "quills"
    if re.search(r"episode|ep[_ ]?\d+", s): return "episode"
    if "full_video"  in s: return "full_video"
    if "notes"       in s: return "notes"
    return None


def ffprobe_duration(path: Path) -> float:
    """Duration in seconds, or NaN on any failure."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             check=True, timeout=60).stdout
        dur = json.loads(out).get("format", {}).get("duration")
        return float(dur) if dur is not None else np.nan
    except Exception:
        return np.nan


# walk the tree
print("scanning directory tree...")
records = []
for path in ROOT.rglob("*"):
    if path.name.startswith("."):
        continue
    rel    = path.relative_to(ROOT)
    suffix = path.suffix.lower()

    if suffix in VIDEO_EXTS:
        ftype = "video"
    elif suffix in DOC_EXTS:
        ftype = "doc"
    elif path.is_dir():
        ftype = "directory"
    else:
        ftype = "other"

    rec = {
        "path":          str(path),
        "relative_path": str(rel),
        "name":          path.name,
        "suffix":        suffix,
        "is_file":       path.is_file(),
        "is_dir":        path.is_dir(),
        "file_type":     ftype,
        "top_level":     get_top_level(rel),
        "subject_id":    parse_subject_id(rel),
        "age_month":     parse_age_month(rel),
        "phase":         parse_phase(rel),
        "task":          parse_task(path.name),
        "task_variant":  parse_task_variant(path.name),
    }
    try:
        stat = path.stat()
        rec["size_bytes"] = stat.st_size
        rec["size_mb"]    = stat.st_size / (1024 ** 2)
    except Exception:
        rec["size_bytes"] = np.nan
        rec["size_mb"]    = np.nan

    records.append(rec)

df = pd.DataFrame(records)
print(f"{len(df):,} items found")


# pull video durations via ffprobe
video_paths = df[df["file_type"] == "video"]["path"].tolist()
print(f"extracting durations for {len(video_paths):,} videos...")

dur_map: dict[str, float] = {}
for p in tqdm(video_paths, desc="ffprobe"):
    dur_map[p] = ffprobe_duration(Path(p))

df["duration_sec"] = df["path"].map(dur_map)
df["duration_min"] = df["duration_sec"] / 60


# annotation files = experimenter session notes, used as candidate labels
# naming convention: {SubjectID}_{Phase}_Notes.docx.pdf  (also raw .docx / .pages)
df["is_annotation"] = (
    df["name"].str.contains(r"notes", case=False, na=False)
    & df["suffix"].isin([".pdf", ".docx", ".pages", ".doc"])
)

# pull phase out of annotation filenames, e.g. 100_E2_Notes.docx.pdf -> "E2"
df["annotation_phase"] = df["name"].str.extract(r'(?<![A-Za-z])(E[123])(?![A-Za-z0-9])', expand=False)


# summary tables

top_summary = (
    df.groupby("top_level")
    .agg(
        n_items      =("path",       "count"),
        n_files      =("is_file",    lambda s: int(s.sum())),
        n_dirs       =("is_dir",     lambda s: int(s.sum())),
        total_size_gb=("size_bytes", lambda s: s.fillna(0).sum() / (1024 ** 3)),
    )
    .sort_values("n_items", ascending=False)
    .reset_index()
)

filetype_summary = (
    df[df["is_file"]]
    .groupby(["top_level", "file_type", "suffix"])
    .size()
    .reset_index(name="count")
    .sort_values(["top_level", "count"], ascending=[True, False])
)

video_summary = (
    df[(df["file_type"] == "video") & df["duration_sec"].notna()]
    .groupby("top_level")
    .agg(
        n_videos_with_duration=("path",         "count"),
        total_hours            =("duration_sec", lambda s: s.sum() / 3600),
        median_min             =("duration_min", "median"),
        min_min                =("duration_min", "min"),
        max_min                =("duration_min", "max"),
    )
    .sort_values("n_videos_with_duration", ascending=False)
    .reset_index()
)

# videos only, known tasks only
task_summary = (
    df[(df["file_type"] == "video") & df["task"].notna()]
    .groupby(["top_level", "task"])
    .size()
    .reset_index(name="count")
    .sort_values(["top_level", "count"], ascending=[True, False])
)

# per subject x phase: which tasks present, which missing
EXPECTED_TASKS_MAP = {
    "6_Months":  EXPECTED_6MO_TASKS,
    "12_Months": EXPECTED_12MO_TASKS,
    "42_Months": EXPECTED_42MO_TASKS,
}

target_video = df[
    df["top_level"].isin(EXPECTED_TASKS_MAP.keys()) &
    (df["file_type"] == "video") &
    df["subject_id"].notna()
]

subject_task_summary = (
    target_video
    .groupby(["top_level", "subject_id", "phase"])["task"]
    .agg(lambda x: sorted(set(t for t in x if pd.notna(t))))
    .reset_index()
    .rename(columns={"task": "tasks_present"})
)

def get_missing(row):
    expected = EXPECTED_TASKS_MAP.get(row["top_level"], set())
    return sorted(expected - set(row["tasks_present"]))

subject_task_summary["missing_tasks"] = subject_task_summary.apply(get_missing, axis=1)
subject_task_summary["n_missing"]     = subject_task_summary["missing_tasks"].apply(len)

# unparsed files = video/doc where task didn't match any pattern
# skip:
#   - date-stamped filenames like "2023-09-06 11-05-50_screen.mp4" (already handled via _screen)
#   - office lockfiles starting with ~$
#   - non-task survey/microbiome paperwork
_DATE_FNAME = re.compile(r"^\d{4}-\d{2}-\d{2}")
_KNOWN_NON_TASK = re.compile(
    r"survey|food.diary|microbiome|gut.survey|gut.microbiome|stool",
    re.IGNORECASE
)

unparsed_mask = (
    df["is_file"]
    & df["file_type"].isin(["video", "doc"])
    & df["task"].isna()
    & ~df["name"].apply(lambda n: bool(_DATE_FNAME.match(n)))
    & ~df["name"].str.startswith("~$")
    & ~df["name"].apply(lambda n: bool(_KNOWN_NON_TASK.search(n)))
)
unparsed_files = (
    df[unparsed_mask][[
        "relative_path", "name", "top_level", "suffix",
        "subject_id", "task", "task_variant",
    ]]
    .sort_values(["top_level", "relative_path"])
)

# annotation files = the main label source
annotation_files = (
    df[df["is_annotation"]][[
        "relative_path", "name", "top_level", "suffix",
        "subject_id", "annotation_phase",
    ]]
    .sort_values(["top_level", "subject_id", "annotation_phase"])
)

# how many subjects have notes per timepoint x phase
annotation_coverage = (
    annotation_files[annotation_files["subject_id"].notna()]
    .groupby(["top_level", "annotation_phase"])
    .agg(
        n_subjects=("subject_id",    "nunique"),
        n_files   =("relative_path", "count"),
    )
    .reset_index()
    .sort_values(["top_level", "annotation_phase"])
)


# write everything out
OUTDIR.mkdir(exist_ok=True, parents=True)

top_summary          .to_csv(OUTDIR / "top_level_summary.csv",     index=False)
filetype_summary     .to_csv(OUTDIR / "filetype_summary.csv",      index=False)
video_summary        .to_csv(OUTDIR / "video_summary.csv",         index=False)
task_summary         .to_csv(OUTDIR / "task_summary.csv",          index=False)
subject_task_summary .to_csv(OUTDIR / "subject_task_summary.csv",  index=False)
unparsed_files       .to_csv(OUTDIR / "unparsed_files.csv",        index=False)
annotation_files     .to_csv(OUTDIR / "annotation_files.csv",      index=False)
annotation_coverage  .to_csv(OUTDIR / "annotation_coverage.csv",   index=False)
df                   .to_csv(OUTDIR / "dataset_inventory_full.csv", index=False)

print(f"\noutputs saved to {OUTDIR.resolve()}")
print(f"total items:       {len(df):,}")
print(f"video files:       {(df['file_type']=='video').sum():,}")
print(f"annotation files:  {df['is_annotation'].sum():,}")
print(f"subjects w/ notes: {annotation_files['subject_id'].nunique():,}")
print(f"unparsed files:    {len(unparsed_files):,}")
print(f"\nannotation coverage:")
print(annotation_coverage.to_string(index=False))