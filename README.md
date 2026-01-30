# Cube Extraction & Alignment from Point Clouds

**MRAC – Student Group Project**

---

## 📋 Overview

This project focuses on extracting a cube-like object from raw point cloud scans, removing the floor, segmenting planar faces, and robustly aligning the cube to a canonical local XYZ coordinate system. The workflow was developed iteratively using Open3D and Python, with a strong emphasis on explainability and debugging through visual feedback.

---

## 🔄 Pipeline Summary

The processing workflow follows these sequential steps:

1. **Floor Detection & Removal** – Isolate the cube from the ground plane
2. **Cube Isolation** – Extract the cube object from the scene
3. **Plane Segmentation** – Identify individual planar faces
4. **Face Labeling** – Classify faces using normal vectors
5. **Robust Orientation Alignment** – Align cube to canonical axes
6. **Final Cube Export** – Save processed geometry

---

## 📁 Folder Structure

```
cube_demo/
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
```

---

## ⚙️ Environment Setup

### Operating System
- Windows 10 / Ubuntu 20.04+

### Python Environment
- **Package Manager:** Conda
- **Python Version:** 3.10
- **Compiler:** Conda default toolchain (MSVC on Windows / GCC on Linux)

**Installation Steps:**

```bash
conda create -n Software2 python=3.10
conda activate Software2
pip install numpy open3d scipy scikit-learn matplotlib
```

### Main Dependencies

| Library | Purpose |
|---------|---------|
| **Open3D** | Point cloud processing |
| **NumPy** | Numerical operations |
| **SciPy** | Spatial mathematics |
| **scikit-learn** | DBSCAN clustering |
| **Matplotlib** | Visualization and debugging |

---

## 🧠 Key Concepts Used

- **RANSAC Plane Detection** – Robust plane fitting with inliers vs outliers
- **DBSCAN Clustering** – Largest cluster extraction
- **Plane Normal Comparison** – Face orientation analysis
- **Oriented Bounding Boxes (OBB)** – Geometric hull computation
- **Normal-Based Axis Alignment** – Canonical orientation recovery

---

## ✅ What Finally Made the Alignment Work

Reliable alignment was only achieved after implementing the following strategy:

1. **Explicit plane (face) extraction** – Rather than relying on global geometry
2. **Normal-based face labeling** – Identifying each face by its orientation
3. **Avoiding pure OBB-based alignment** – OBB proved insufficient for canonical orientation
4. **Using dominant planar normals** – To define the local XYZ axes accurately

**Final implementation:** `12_align_planes_robust_v2.py`

---

## 🚀 Running the Pipeline

### Execution Order

```bash
# Step 1: Isolate the cube from the raw scan
python 02_isolate_cube_v2.py

# Step 2: Export cube-only point cloud
python 10_export_cube_only_v2.py

# Step 3: Align cube to canonical axes
python 12_align_planes_robust_v2.py
```

Each script exports intermediate `.ply` files for inspection and debugging.

---

## 🔗 Project Links

### Development Documentation

- **ChatGPT Discussion** (process & debugging):  
  [View conversation](https://chatgpt.com/share/696fc076-d1f4-800f-8bfd-f76d6471dd11)

- **Claude Discussion** (alternative reasoning):  
  [View conversation](https://claude.ai/share/d72d2f0c-528a-4dd7-96d2-82b9f2c0cb25)

- **Miro Board** (visual reasoning & pipeline):  
  [View board](https://miro.com/welcomeonboard/QjFOT0F1WnhOS3ZRYmR4K0YxalhrYk54aytpUjkvQkhVZlFPVndMcTRKL3dueVpqZUFnNGs0QSs2emxrVDVuejdQOVNPQjZNN3FXM3ByanhFVzJ5WnkyczdId2FwLzJEZ3VuRWo5MFNuaFpRR3hORWxvOFBndVF4RmlKenVnTlR3VHhHVHd5UWtSM1BidUtUYmxycDRnPT0hdjE=?share_link_id=804063472821)

---

## 👥 Team

**Core Development:**
- Leonard
- XIO
- Jenny

**Testing & Evaluation:**
- Group Members

---

## 📝 Notes

This project demonstrates a complete pipeline for geometric extraction and alignment from noisy 3D scan data, with emphasis on robustness and visual verification at each processing stage.
