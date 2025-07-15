# 📌 Pneumonia Detection using VGG16 and Transfer Learning

This project aims to classify chest X-ray images into two categories — **Normal** and **Pneumonia** — using a deep learning approach powered by transfer learning with the **VGG16** architecture.

---

## 📁 Project Overview

- **Goal:** Binary classification of chest X-rays (Pneumonia vs. Normal)
- **Dataset:** NIH Chest X-ray Dataset (sourced via Kaggle)
- **Approach:** Transfer learning with VGG16 (pre-trained on ImageNet)
- **Frameworks Used:** TensorFlow, Keras
- **Environment:** Jupyter Notebook (VS Code)

---

## 🧪 Data Preprocessing

- All images resized to **128×128**
- **Data Augmentation** applied to enhance generalization:
  - Rotation
  - Width & Height Shift
  - Zoom
  - Horizontal Flip
- **Dataset Split**:
  - Training: 90%
  - Validation: 10%
  - Testing: Separate folder

---

## 🧠 Model Architecture

- **Base Model:** VGG16 (frozen weights from ImageNet)
- **Custom Layers Added:**
  - `GlobalAveragePooling2D`
  - Fully connected layers: 256 → 128 → 64
  - `BatchNormalization`, `Dropout` layers for regularization
  - Final layer with `Sigmoid` activation for binary classification
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

---

## ⚙️ Training Strategy

- **Transfer Learning:** VGG16 base layers frozen; only custom top layers trained
- **Callbacks Used:**
  - `EarlyStopping` – stops training if validation loss stops improving
  - `ReduceLROnPlateau` – reduces learning rate when validation loss plateaus
  - `ModelCheckpoint` – saves best model weights based on validation accuracy

---

## 📈 Results

- **Epochs Trained:** 20
- **Best Validation Accuracy:** ~92%
- **Test Accuracy:** ~90.5%
- **Model Checkpoint Saved:**  
  `model_weights_02/vgg_model_new_02.h5`

---

## 💡 Conclusion

By leveraging VGG16 and transfer learning, this model achieved strong classification performance on chest X-ray images. The combination of augmentation, regularization, and training callbacks contributed significantly to its effectiveness.

---

## 🚀 Getting Started

### 📦 Install Dependencies

```bash
pip install -r requirements.txt

Authors:
Joe Cansio Fernandes
Shreyas Prabhukanolkar
