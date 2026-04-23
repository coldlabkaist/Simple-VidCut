# Simple VidCut

Simple VidCut is a desktop app for trimming local video files with frame/time controls, bookmarks, and image adjustments.

## Features
- Video playback and scrubbing
- Trim parameter modes:
  - `Start + Duration`
  - `Duration + End`
  - `Start + End` (default)
- Export modes:
  - `accurate (re-encode)` for precise cut points
  - `fast (stream copy)` for faster export (keyframe-aligned)
- Single export: `Save Current Video`
- Batch export: `Save Videos...` (multi-select with Ctrl/Shift/Ctrl+A)
- Image adjustments: contrast, brightness, saturation (applied to preview and export)

## Requirements
- Conda (Miniconda or Anaconda)
- Python (3.9+ recommended)
- FFmpeg (required)

## Installation
### 1) Clone the project
```
git clone https://github.com/coldlabkaist/Simple-VidCut.git
cd Simple-VidCut
```

### 2) Create and activate the conda environment
```powershell
conda env create -f environment.yml
conda activate simplevidcut
```

### 3) Verify FFmpeg
```
ffmpeg -version
```

## Run
```
python SimpleVidCut.py
```