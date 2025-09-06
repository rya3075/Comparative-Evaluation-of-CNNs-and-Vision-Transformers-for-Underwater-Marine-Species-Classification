# Underwater Animal Classification using Transformers and CNNs

## 📌 Project Overview
Underwater biodiversity monitoring faces critical challenges due to **image degradation** from light absorption and scattering, making automated classification systems essential for **marine conservation**.  
This project systematically evaluates **transformer architectures** against **convolutional neural networks (CNNs)** for **underwater animal classification** using a **23-species marine dataset (13,711 images)**.

We implement a **unified transfer learning framework** with:
- Selective fine-tuning  
- **Focal Loss** for handling class imbalance  
- **Differential learning rates** for optimized training  

## 🧪 Models Evaluated
- **ResNet-50 (CNN baseline)**
- **Vision Transformer (ViT-B/16)**
- **Swin Transformer (Swin-B)**

## 📊 Results
| Model            | Accuracy (%) |
|------------------|--------------|
| ResNet-50        | 85.84        |
| ViT-B/16         | 90.37        |
| Swin Transformer | **92.06**    |

- **Swin Transformer** achieved the best performance, showing that **hierarchical attention mechanisms** are particularly effective for morphologically complex species.  
- **Statistical validation** was performed using:
  - **McNemar’s test**  
  - **5-fold stratified cross-validation**  
  confirming significant differences among models.  

## 🔑 Key Insights
- **Transformers outperform CNNs** in underwater image classification.  
- **Self-attention mechanisms** help mitigate underwater imaging challenges (light scattering, color loss).  
- Despite higher computational demands, transformers set **new state-of-the-art performance** for underwater biodiversity monitoring.  

## 📂 Dataset
- **Marine dataset** with **23 species** and **13,711 labeled images**.  
- Images include varying levels of noise, blur, and color distortion common in underwater environments.  

*(Dataset source not included here for licensing reasons — please add your dataset link if publicly available.)*

## ⚙️ Methodology
1. Preprocessing with data augmentation (color jitter, random crop, flipping).  
2. Transfer learning applied to each model.  
3. Selective fine-tuning with layer-wise learning rates.  
4. Class imbalance handled via **Focal Loss**.  
5. Evaluation using accuracy, precision, recall, F1-score, and statistical tests.  

## 🚀 Technologies Used
- **Python 3.10+**
- **PyTorch**
- **Transformers (HuggingFace)**
- **scikit-learn**
- **NumPy, Pandas, Matplotlib, Seaborn**

## 📈 Applications
- **Automated marine species monitoring**  
- **Biodiversity conservation strategies**  
- **Support for marine research organizations**  

## 🧑‍💻 Authors
- Aarya Nandurkar  
