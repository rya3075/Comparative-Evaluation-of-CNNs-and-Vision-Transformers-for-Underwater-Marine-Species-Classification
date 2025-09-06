# Underwater Animal Classification using Transformers and CNNs

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Available-brightgreen.svg)](#citation)

## 📌 Overview

Underwater biodiversity monitoring faces critical challenges due to **image degradation** from light absorption and scattering in marine environments. This project addresses these challenges by developing and evaluating automated classification systems for marine conservation applications.

We present a systematic comparison of **transformer architectures** against **convolutional neural networks (CNNs)** for underwater animal classification using a comprehensive 23-species marine dataset containing 13,711 images. Our unified transfer learning framework demonstrates that hierarchical attention mechanisms significantly outperform traditional CNN approaches in underwater imaging scenarios.

## 🎯 Key Contributions

- **Comprehensive evaluation** of transformer vs CNN architectures for underwater image classification
- **Novel application** of Swin Transformer to marine biodiversity monitoring
- **Unified framework** incorporating focal loss and differential learning rates for underwater scenarios
- **Statistical validation** using McNemar's test and cross-validation
- **State-of-the-art results** achieving 92.06% accuracy on underwater species classification

## 🧪 Model Architecture Comparison

### Models Evaluated

| Model | Type | Parameters | Key Features |
|-------|------|------------|--------------|
| **ResNet-50** | CNN | 25.6M | Deep residual connections, established baseline |
| **Vision Transformer (ViT-B/16)** | Transformer | 86.6M | Self-attention mechanisms, global context modeling |
| **Swin Transformer (Swin-B)** | Hierarchical Transformer | 88.0M | Shifted windows, multi-scale feature extraction |

### Performance Results

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Inference Time (ms) |
|-------|--------------|---------------|------------|--------------|-------------------|
| ResNet-50 | 85.84 | 84.72 | 85.84 | 84.91 | 12.3 |
| ViT-B/16 | 90.37 | 89.45 | 90.37 | 89.83 | 28.7 |
| **Swin Transformer** | **92.06** | **91.34** | **92.06** | **91.65** | 31.2 |

## 📊 Dataset

### Dataset Characteristics
- **Species Count**: 23 marine species
- **Total Images**: 13,711 labeled images
- **Image Resolution**: 224×224 pixels (after preprocessing)
- **Class Distribution**: Imbalanced (handled via Focal Loss)
- **Environmental Conditions**: Varying water clarity, lighting, and depth

### Species Categories
The dataset includes diverse marine life categories:
- **Fish Species**: Multiple families including tropical and deep-sea varieties
- **Marine Mammals**: Dolphins, whales, seals
- **Invertebrates**: Jellyfish, octopi, sea turtles
- **Coral Reef Species**: Various reef-associated animals

### Data Challenges
- Light absorption and scattering effects
- Color distortion in deeper waters
- Motion blur from animal movement
- Varying image quality and resolution
- Occlusion and camouflage patterns

## ⚙️ Methodology

### 1. Data Preprocessing Pipeline
```python
# Data augmentation strategy
transforms = [
    RandomResizedCrop(224),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
```

### 2. Transfer Learning Framework
- **Backbone Initialization**: Pre-trained weights from ImageNet
- **Selective Fine-tuning**: Layer-wise learning rate scheduling
- **Gradient Accumulation**: Batch size optimization for memory efficiency

### 3. Loss Function and Optimization
```python
# Focal Loss for class imbalance
focal_loss = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')

# Differential learning rates
optimizer_params = [
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
]
```

### 4. Evaluation Protocol
- **Cross-validation**: 5-fold stratified cross-validation
- **Statistical Testing**: McNemar's test for model comparison
- **Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Confidence Intervals**: Bootstrap sampling (n=1000)

## 🔬 Experimental Results

### Statistical Significance Testing
McNemar's test results comparing model pairs:

| Model Pair | χ² Statistic | p-value | Significance |
|------------|--------------|---------|--------------|
| Swin vs ViT | 15.23 | < 0.001 | *** |
| Swin vs ResNet | 47.84 | < 0.001 | *** |
| ViT vs ResNet | 22.16 | < 0.001 | *** |

### Cross-Validation Results
```
Swin Transformer: 92.06% ± 1.8%
ViT-B/16:        90.37% ± 2.1%
ResNet-50:       85.84% ± 2.3%
```

### Ablation Studies
| Component | Accuracy Impact |
|-----------|-----------------|
| Focal Loss | +3.2% |
| Data Augmentation | +4.7% |
| Differential LR | +1.9% |
| Pre-training | +8.3% |



## 🔍 Key Insights and Analysis

### Why Transformers Excel in Underwater Scenarios

1. **Global Context Modeling**: Self-attention mechanisms capture long-range dependencies crucial for identifying partially occluded marine animals

2. **Robustness to Distortion**: Transformers show superior resilience to color distortion and lighting variations common in underwater environments

3. **Hierarchical Processing**: Swin Transformer's windowed attention effectively handles multi-scale features from microscopic to large marine species

4. **Adaptive Feature Learning**: Attention weights automatically focus on discriminative regions despite underwater imaging artifacts

### Computational Considerations

| Model | Training Time (h) | GPU Memory (GB) | FLOPs (G) |
|-------|-------------------|-----------------|-----------|
| ResNet-50 | 2.3 | 6.2 | 4.1 |
| ViT-B/16 | 4.1 | 12.8 | 17.6 |
| Swin Transformer | 4.7 | 14.3 | 15.4 |

## 📈 Applications and Impact

### Marine Conservation Applications
- **Real-time species monitoring** in protected marine areas
- **Biodiversity assessment** for environmental impact studies
- **Automated species counting** for population studies
- **Citizen science** tools for marine researchers

### Deployment Scenarios
- **Underwater ROV systems** for deep-sea exploration
- **Marine surveillance networks** for conservation monitoring
- **Educational platforms** for marine biology learning
- **Research vessel integration** for field studies

## 🔮 Future Work

### Immediate Extensions
- [ ] **Multi-modal fusion** incorporating acoustic and environmental data
- [ ] **Semi-supervised learning** for leveraging unlabeled underwater imagery
- [ ] **Edge deployment optimization** for real-time underwater systems
- [ ] **Temporal modeling** for video-based species tracking

### Long-term Research Directions
- [ ] **Domain adaptation** across different marine environments
- [ ] **Few-shot learning** for rare species identification
- [ ] **Explainable AI** for marine biologist collaboration
- [ ] **Federated learning** for multi-institution data sharing

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📞 Contact

**Aarya Nandurkar**
- Email: [ryanandurkar30@gmail.com](mailto:ryanandurkar30@gmail.com)


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Marine Instrumentation Division, CSIR-NIO, Goa

---

<div align="center">
  <strong>🌊 Advancing Marine Conservation through AI 🐠</strong>
</div>
