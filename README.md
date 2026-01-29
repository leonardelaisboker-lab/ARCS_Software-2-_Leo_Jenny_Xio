# Cube Extraction & Alignment from Point Clouds  
**MRAC – Student Group Project**

## Overview
This project focuses on extracting a cube-like object from raw point cloud scans, removing the floor, segmenting planar faces, and robustly aligning the cube to a canonical local XYZ coordinate system. The workflow was developed iteratively using Open3D and Python, with a strong emphasis on explainability and debugging through visual feedback.

---

## Pipeline Summary
1. Floor detection & removal  
2. Cube isolation  
3. Plane segmentation (faces)  
4. Face labeling via normals  
5. Robust orientation alignment  
6. Final cube export  

---

## Folder Structure
ube_demo/
│
├── 01_measure_cube.py
├── 02_isolate_cube.py
├── 02_isolate_cube_v2.py
├── 03_measure_cube_obb.py
├── 06_remove_floor.py
├── 06b_remove_floor.py
├── 10_export_cube_only.py
├── 10_export_cube_only_v2.py
├── 11_cube_faces_colorize.py
├── 12_align_planes_robust.py
├── 12_align_planes_robust_v2.py
│
├── scan_raw.ply
├── scan_floor_aligned.ply
├── cube_only.ply
└── README.md

---

## Environment Setup

### Operating System
- Windows 10 / Ubuntu 20.04+

### Python Environment
- Conda
- Python 3.10  
- Compiler: Conda default toolchain (MSVC on Windows / GCC on Linux)

```bash
conda create -n Software2 python=3.10
conda activate Software2
pip install numpy open3d scipy scikit-learn matplotlib
Main Dependencies

Open3D – point cloud processing

NumPy – numerical operations

SciPy – spatial mathematics

scikit-learn – DBSCAN clustering

Matplotlib – visualization and debugging

Key Concepts Used

RANSAC plane detection

Inliers vs outliers

DBSCAN clustering (largest cluster extraction)

Plane normal comparison

Oriented Bounding Boxes (OBB)

Normal-based axis alignment

What Finally Made the Alignment Work

Reliable alignment was only achieved after:

Explicit plane (face) extraction

Normal-based face labeling

Avoiding pure OBB-based alignment

Using dominant planar normals to define the local XYZ axes

This final logic is implemented in:

12_align_planes_robust_v2.py

Running the Pipeline

Example execution order:

python 02_isolate_cube_v2.py
python 10_export_cube_only_v2.py
python 12_align_planes_robust_v2.py


Each script exports intermediate .ply files for inspection and debugging.

Project Links

ChatGPT discussion (process & debugging):
https://your-chatgpt-link-here

Claude discussion (alternative reasoning):
https://your-claude-link-here

Miro board (visual reasoning & pipeline):
https://your-miro-link-here

Authors

Leonard
XIO
Jenny 

Group Members – testing, evaluation, presentation
