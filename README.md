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


conda create -n Software2 python=3.10
conda activate Software2
pip install numpy open3d scipy scikit-learn matplotlib

### Main Dependencies

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

## Example execution order:

python 02_isolate_cube_v2.py
python 10_export_cube_only_v2.py
python 12_align_planes_robust_v2.py


Each script exports intermediate .ply files for inspection and debugging.

### Project Links

ChatGPT discussion [(process & debugging):](https://chatgpt.com/share/696fc076-d1f4-800f-8bfd-f76d6471dd11)


Claude discussion (alternative reasoning):https://claude.ai/share/d72d2f0c-528a-4dd7-96d2-82b9f2c0cb25


Miro board (visual reasoning & pipeline): [https://miro.com/app/board/uXjVGMOQ9k4=/](https://miro.com/welcomeonboard/QjFOT0F1WnhOS3ZRYmR4K0YxalhrYk54aytpUjkvQkhVZlFPVndMcTRKL3dueVpqZUFnNGs0QSs2emxrVDVuejdQOVNPQjZNN3FXM3ByanhFVzJ5WnkyczdId2FwLzJEZ3VuRWo5MFNuaFpRR3hORWxvOFBndVF4RmlKenVnTlR3VHhHVHd5UWtSM1BidUtUYmxycDRnPT0hdjE=?share_link_id=804063472821)
https://your-miro-link-here

Authors

Leonard
XIO
Jenny 

Group Members – testing, evaluation, presentation
