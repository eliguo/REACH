# REACH: Remote Assessment of Child Health

Code for the MS Capstone project *Behavioral Phenoscreening: Remote Assessment of Child Mental Health (REACH)*, NYU Center for Data Science, Spring 2026.

REACH is a multi-modal pipeline for automated behavioral analysis of parent-recorded infant–caregiver interaction videos. The repo is split into three parts:

- `action_recognition/` — pose estimation and skeleton-based action recognition benchmarks on InfActPrimitive (infant) and NTU RGB+D (adult), plus the InfoGCN baseline we explored.
- `cope/` — dataset audit and inventory tooling for the COPE longitudinal video collection (4,163 videos across 6 / 12 / 42 month timepoints).
- `mediapipe/` — holistic landmark detection experiments (face + body + hands).

Each subdirectory has its own README with setup and reproduction notes.

## Mentors

Prof. Yiqiu Shen, Prof. Lauren Shuffrey (NYU Langone).

## Authors

Eli Guo, Zijin Hu, Iris Wu, Tim Zhou.
