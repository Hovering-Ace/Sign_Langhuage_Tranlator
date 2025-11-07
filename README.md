# 👐 SignSpeak — A–Z Sign Language Translator (with Blank Token Support)

**SignSpeak** is an AI-powered sign language translator that recognizes hand gestures for **A–Z alphabets** and a **blank/pause symbol** for spacing and natural sentence segmentation.  
It uses computer vision and deep learning to convert video input into real-time English text — designed for research, accessibility, and educational applications.

---

## 🚀 Overview

This project provides a complete end-to-end pipeline for **sign-to-text translation**, from video preprocessing to model training and live inference.  
By incorporating a **blank token**, the system learns to detect pauses between gestures, making continuous signing more natural and readable.

---

## 🧠 Key Features

- **27-Class Recognition:** Supports A–Z alphabets plus a blank/pause token for seamless sentence flow.  
- **Pose-Based Vision Pipeline:** Uses **Mediapipe** and **OpenCV** for robust hand and landmark extraction.  
- **Deep Learning Backbone:** Hybrid **CNN + BiLSTM / Transformer** architecture for spatiotemporal modeling.  
- **CTC Loss for Alignment-Free Training:** Enables training without frame-level labels, relying on blank token transitions.  
- **Class Balancing:** Integrates **SMOTE** and data augmentation to improve accuracy on underrepresented signs.  
- **Real-Time Web App:** Interactive **Flask/Streamlit** demo for webcam-based sign recognition with smooth **GSAP animations**.  
- **Detailed Evaluation:** Token Error Rate (TER), letter-wise accuracy, confusion matrix, and F1 scores.

---

## 🧩 Tech Stack

`Python` • `PyTorch` • `TensorFlow` • `OpenCV` • `Mediapipe` • `NumPy` • `Pandas` • `scikit-learn` • `Streamlit` / `Flask`

---

## 🚀 output
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/ccab04bd-84c7-42b3-9f50-46756a02ca56" />
##fig1
<img width="1365" height="767" alt="image" src="https://github.com/user-attachments/assets/ef765e68-6b53-4197-98cf-b1976ee08e66" />
##fig2
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/30f5ec67-0f7c-4e9d-9a8d-3d7554b4dd4d" />
##fig3


