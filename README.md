# 🌺 FloraScan: AI Botanical Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Status-Active-success)

**FloraScan** is an end-to-end Deep Learning application capable of identifying 102 different species of flowers from images. It leverages **Transfer Learning (ResNet34)** to handle Fine-Grained Visual Categorization and provides a user-friendly web interface via **Streamlit**.

---

## 📖 Project Rationale

### 1. The Challenge: Fine-Grained Visual Categorization (FGVC)
Unlike standard classification tasks (e.g., Cats vs. Dogs), identifying flower species is a **Fine-Grained** problem. Distinct species often share very similar structures, while a single species can vary drastically based on lighting and age. This project demonstrates the capacity of Deep Learning to discern subtle textural details that even humans find difficult.

### 2. Technical Approach: Transfer Learning
Training a deep CNN from scratch requires massive datasets. FloraScan utilizes **Transfer Learning** with a **ResNet34** architecture pre-trained on ImageNet. By freezing early layers and fine-tuning the deeper layers, the model effectively extracts high-level features (edges, textures) while learning the specific biological nuances of the **Oxford Flowers102** dataset.

### 3. Real-World Application
This project bridges the gap between research and product. By deploying the model with **Streamlit** and **Ngrok**, it transforms a static notebook into an interactive digital tool for botanists and nature enthusiasts.

---

## ✨ Key Features

* **102 Flower Classes:** Trained on the challenging Oxford Flowers102 dataset.
* **Deep Learning Engine:** Powered by PyTorch and ResNet34.
* **Robust Preprocessing:** Implements Data Augmentation (Rotation, Color Jitter, Scaling) to handle real-world image noise.
* **Interactive UI:** Web-based interface built with Streamlit.
* **Confidence Metrics:** Visualizes top-3 prediction probabilities using Plotly charts.
* **Cloud Ready:** Designed to run on Google Colab with ephemeral deployment via Ngrok.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Core Framework:** PyTorch, Torchvision
* **Frontend:** Streamlit
* **Visualization:** Plotly, Matplotlib
* **Deployment:** Ngrok
* **Dataset:** Oxford Flowers102

---

## 🚀 How to Run (Google Colab)

This project is optimized for Google Colab (Free Tier GPU).

1.  **Clone/Open the Notebook:** Upload the provided `.ipynb` file to Google Colab.
2.  **Set Runtime:** Go to `Runtime` > `Change runtime type` > Select **T4 GPU**.
3.  **Install Dependencies:**
    ```python
    !pip install torch torchvision streamlit pyngrok
    ```
4.  **Add Ngrok Token:**
    * Sign up at [ngrok.com](https://ngrok.com).
    * Copy your Authtoken.
    * Replace `YOUR_NGROK_AUTH_TOKEN` in the deployment cell.
5.  **Run All Cells:** The final cell will generate a public URL (e.g., `https://xxxx.ngrok-free.app`).


