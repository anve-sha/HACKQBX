<div align="center">

# 🌿 Offroad Scene Segmentation
**Next-Generation Environmental Understanding using DINOv2 & Vision APIs**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](#)
[![DINOv2](https://img.shields.io/badge/DINOv2-Meta_AI-blue?style=for-the-badge)](#)
[![Hackathon](https://img.shields.io/badge/Hackathon-Project-success?style=for-the-badge)](#)

</div>

---

## 🎯 Project Overview
Understanding unstructured natural environments is notoriously difficult for traditional computer vision systems. **Offroad Scene Segmentation** bridges that gap by providing a high-performance semantic segmentation engine capable of classifying natural landscapes—trees, dry bushes, rocks, logs, and more—into 10 distinct environmental classes. 

This project goes beyond just a machine learning notebook. I built a **full-stack, production-ready system** wrapping the cutting-edge PyTorch model inside an optimized FastAPI backend and serving it via a seamless, glassmorphism-styled Web UI.

---

## ✨ Key Features
- 🤖 **Foundation Model Integration:** Leverages Meta's DINOv2 self-supervised vision transformer as an incredibly powerful feature extractor.
- ⚡ **Lightweight ConvNeXt Head:** A custom, highly efficient segmentation head designed specifically to aggregate DINO patch-tokens.
- 🎨 **Full-Stack Glassmorphism UI:** A beautiful, animated frontend interface to upload images, stream them to the API, and dynamically visualize predicted mask boundaries.
- 🚀 **Production-Ready API Endpoint:** Decoupled FastAPI backend optimized with CPU-fallbacks for seamless Render.com deployments, eliminating CUDA bloat.

---

## 🏗️ Model Architecture

To achieve accurate environmental feature bounding on a constrained time and data budget, the architecture utilizes a freeze-and-train methodology:

1. **Backbone (`DINOv2-vits14`):** Frozen self-supervised vision transformer. DINOv2 naturally understands depth, contours, and object boundaries without fine-tuning, outputting rich patch-token embeddings.
2. **Segmentation Head (`ConvNeXt-style`):** Uses an un-frozen `stem -> block -> classifier` pipeline.
   - Initial `7x7` Depthwise Convolution stem.
   - Processing block utilizing GELU activations and `1x1` point-wise combinations.
   - Outputs logit masks mapped perfectly to 10 environmental target classes.

---

## 📊 Training Details & Results

The model was trained entirely on a custom Offroad dataset with mapped pixels ranging from `0` (Sky) to `10000` (Background clutter). 

### ⚙️ Hyperparameters
* **Epochs:** `10`
* **Batch Size:** `2`
* **Learning Rate:** `1e-4`
* **Optimizer:** SGD (Momentum = 0.9)
* **Image Dimensions:** Resized heavily to `960x540` ensuring structural integrity during multi-scaling.

### 🏆 Final Evaluation Metrics
Despite a constrained hackathon timeframe and complex 10-class unstructured nature classes, the model achieved strong initial clustering capability:

| Metric | Score | Note |
|---|---|---|
| **Best Pixel Accuracy** | `70.16%` | *Achieved on Epoch 10* |
| **Best Dice Score (F1)** | `43.83%` | *Achieved on Epoch 10* |
| **Best Mean IoU** | `29.35%` | *Achieved on Epoch 10* |
| **Lowest Val Loss** | `0.8181` | *Achieved on Epoch 10* |

<div align="center">
  <img src="train_stats/all_metrics_curves.png" alt="Training Graphs" width="600" />
</div>

---

## 💻 Tech Stack
- **AI / ML Layer:** `PyTorch`, `TorchVision`, `Meta DINOv2`
- **Data Engineering:** `NumPy`, `OpenCV-Headless`, `Pillow`
- **Backend Infrastructure:** `FastAPI`, `Uvicorn`, `Python-Multipart`
- **Frontend / UI:** Vanilla `HTML5`, `CSS3` (Glassmorphism Design), `ES6 JavaScript` (Fetch API)

---

## 📂 Project Structure

```text
├── train_segmentation.py        # Model training script
├── train_stats/                 # Dynamic loss/IoU graph outputs
├── project/
│   ├── backend/
│   │   ├── app.py               # FastAPI PyTorch ingestion server
│   │   └── requirements.txt     # Dependency graph (CPU-Optimized)
│   ├── frontend/
│   │   ├── index.html           # Structure 
│   │   ├── style.css            # Styling
│   │   └── script.js            # Dynamic API integrations
```

---

## 🚀 How to Run Locally

### 1️⃣ Start the Backend API
```bash
cd project/backend/
pip install -r requirements.txt
python app.py
```
*Your model begins serving predictions on `http://127.0.0.1:8000`*

### 2️⃣ Launch the UI
Because the UI is completely decoupled and fully static, you can simply spin up a temporary Python server:
```bash
cd project/frontend/
python -m http.server 3000
```
*Visit `http://localhost:3000` to interact with the Semantic Studio!*

---

## 🧗 Challenges & Solutions
1. **OOM (Out-Of-Memory) Constraints on Cloud Deployment:**
   - *Challenge:* Free-tier deployment platforms like Render constrain memory to 512MB, but PyTorch + CUDA binaries sum up to over 1GB.
   - *Solution:* Engineered the `requirements.txt` to strictly pull the `+cpu` Python wheel, immediately stripping hundreds of megabytes of unwanted GPU driver binaries allowing successful free deployments.
2. **Transforming High-Resolution Unstructured Data:**
   - *Challenge:* Forests and bushes have chaotic borders causing extreme loss spikes in typical ResNet backbones.
   - *Solution:* Anchored the backbone to Meta's DINOv2 transformer sequence which was pre-trained extensively on unstructured data, passing mathematically dense representations into a much smaller, controllable ConvNeXt head.

---

## 🔭 Future Improvements
* Set up a proper CI/CD Github Actions pipeline.
* Augment the initial 10 Epochs to 50 Epochs leveraging dynamic Learning Rate Schedulers (`ReduceLROnPlateau`).
* Implement bounding box aggregations over the segmentation maps to identify exact dimensions of ground-debris boundaries.

---
<div align="center">
<i>Built to solve complex vision problems with elegant engineering.</i>
</div>
