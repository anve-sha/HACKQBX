<div align="center">

# 🌿 AI-Based Offroad Segmentation System
**Next-Generation Environmental Understanding using DINOv2 & Vision APIs**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](#)
[![HTML/CSS/JS](https://img.shields.io/badge/Frontend-Vanilla-orange?style=for-the-badge)](#)
[![HackQBX](https://img.shields.io/badge/Hackathon-HackQBX-success?style=for-the-badge)](#)

</div>

---

## 🎯 Project Overview
Understanding unstructured natural environments is notoriously difficult for traditional computer vision systems. **AI-Based Offroad Segmentation System** bridges that gap by providing a high-performance semantic segmentation engine capable of classifying natural landscapes—trees, dry bushes, rocks, logs, and more—into 10 distinct environmental classes. 

This project goes beyond a simple ML script. I built a **full-stack system** wrapping the cutting-edge PyTorch model inside an optimized FastAPI backend and serving it via a seamless, glassmorphism-styled Web UI deployed on Netlify.

---

## 🚨 Important Note
> **⚠️ Due to hardware limitations and large model size, the backend is recommended to run locally.**
> While the frontend is deployed on Netlify, the PyTorch segmentation model requires significant computational resources that exceed free-tier cloud hosting limits. Please follow the instructions below to run the backend on your own machine.

---

## 🛑 Problem Statement
Off-road autonomous navigation lacks clear road boundaries and predictable structures. Typical vision models fail at separating complex foliage, uneven terrain, and natural debris. A reliable perception system is necessary to semantically segment environmental elements for safe off-road traversability.

---

## 💡 Solution Approach
To achieve accurate environmental feature bounding on a constrained boundary, the architecture utilizes a freeze-and-train methodology:
1. **Backbone (`DINOv2-vits14`):** Frozen self-supervised vision transformer by Meta. DINOv2 naturally understands depth, contours, and object boundaries without fine-tuning.
2. **Segmentation Head (`ConvNeXt-style`):** A custom, highly efficient segmentation head to map patch-tokens to 10 environmental classes. 
3. **Interactive UI:** A user-friendly web interface allowing instant uploads and real-time segmentation map visualization.

---

## 💻 Tech Stack
- **Frontend:** `HTML5`, `CSS3` (Glassmorphism), `JavaScript` (Deployed on **Netlify**)
- **Backend:** `FastAPI`, `Uvicorn` (Runs locally)
- **AI / ML Model:** `PyTorch`, `Meta DINOv2`, `TorchVision`
- **Data Processing:** `NumPy`, `OpenCV-Headless`, `Pillow`

---

## 🚀 Run Backend Locally
Follow these step-by-step instructions to spin up the PyTorch server on your machine:

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd <repo-folder>
   ```

2. **Navigate to the backend directory:**
   ```bash
   cd project/backend
   ```

3. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

4. **Activate the environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

5. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the FastAPI server:**
   ```bash
   uvicorn app:app --reload
   ```

7. **Verify it's running:**
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🔌 Frontend Connection
If the backend is running locally, ensure the frontend points to your local server. 
Update the API URL in `script.js` (inside `project/frontend/` or your Netlify deployed code):
```javascript
fetch("http://127.0.0.1:8000/predict")
```
DEPLOYED LINK FOR FRONTEND-:https://hackqbx.netlify.app
---

## 👨‍⚖️ How to Use (For Judges)
Here are the precise steps to test the full system:

1. **Open the Netlify Frontend Link:** Go to our live deployed website (URL).
2. **Run Backend Locally:** Follow the "Run Backend Locally" steps above to start the FastAPI server on your machine.
3. **Upload Image:** Click the upload module on the frontend and select an offroad natural image.
4. **View Output:** The system will process the image through the local backend and stream the segmented visual output along with IoU, Dice Score, and Accuracy metrics back to the UI!

---

## 📊 Results (IoU score)
Despite complex 10-class unstructured nature classes, the model achieved strong initial clustering capability:

| Metric | Score | Note |
|---|---|---|
| **Best Pixel Accuracy** | `70.16%` | *Achieved on Epoch 10* |
| **Best Dice Score (F1)** | `43.83%` | *Achieved on Epoch 10* |
| **Best Mean IoU** | `29.35%` | *Achieved on Epoch 10* |

*(Evaluations achieved on a custom Offroad dataset for 10 epochs).*
<br>
<div align="center">
  <img src="train_stats/all_metrics_curves.png" alt="Training Graphs" width="500" />
</div>

---

## 🧗 Challenges Faced
1. **OOM (Out-Of-Memory) Constraints on Cloud:** Large ML models easily exceed the 512MB RAM limits of free tiers (like Render). We solved this by decoupling the FastAPI to run locally while keeping the UI accessible on the cloud.
2. **Complex Feature Boundaries:** Forests and bushes have chaotic borders. We overcame this by using the DINOv2 self-supervised transformer which excels at boundary detection in unstructured data.

---

## 🎉 Conclusion
The **AI-Based Offroad Segmentation System** vividly demonstrates the feasibility of combining lightweight UI dashboards with thick, compute-heavy AI backends running entirely locally. It lays the groundwork to enable future rovers and self-driving platforms to navigate untamed environments safely.

---

## ✨ Credits
A huge thank you to the following individuals for their incredible contributions and guidance throughout this project:
- **Ridhi didi** – For invaluable guidance and mentorship.
- **Arohi Verma** – Core contributor.
- **Harshita** – Core contributor.

---

## 🙌 Acknowledgement
Special thanks to the amazing communities and organizations that made this possible:
- 🏆 **HackQBX** 
- 💻 **DevQBX**
- 🛠️ **SiteCraft**

---
<div align="center">
<i>Built with ❤️ to solve complex vision problems.</i>
</div>
