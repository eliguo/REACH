# Audio Analysis for Arm Restraint Experiment

This repository contains the audio analysis workflow developed for the capstone project. The pipeline focuses on extracting audio from video recordings, generating transcript-based temporal references, segmenting the arm restraint task into annotated phases, and comparing MFCC-based acoustic features with annotated negative affect scores.

## Repository Contents

- `audio_extract.ipynb`  
  Extracts WAV audio from original video recordings and generates basic audio outputs such as spectrograms.

- `transcript.ipynb`  
  Runs the speech-to-text pipeline and produces timestamped transcript segments that can be used as approximate temporal references for task boundaries.

- `mfcc_analysis_example.ipynb`  
  Extracts `restrain1` and `restrain2` segments, computes MFCC features, visualizes feature-score relationships, and evaluates agreement between MFCC features and annotated negative affect scores.

- `requirements.txt`  
  Python dependencies used in this project.

## Workflow Overview

The analysis pipeline consists of four main stages:

1. Convert original video recordings into standalone WAV audio files  
2. Generate transcript-based temporal references using speech-to-text  
3. Extract annotated `restrain1` and `restrain2` audio segments  
4. Compute MFCC features and compare them with annotated negative affect scores  

## MFCC Analysis

For each arm restraint segment, 13 MFCC mean features are extracted and compared against annotated negative affect scores. In the current pilot analysis, the most promising features were:

- `mfcc_9_mean`
- `mfcc_2_mean`
- `mfcc_6_mean`

These features showed the strongest agreement with annotated scores in the preliminary sample.

## Installation

Install Python dependencies with:

```bash
pip install -r requirements.txt
