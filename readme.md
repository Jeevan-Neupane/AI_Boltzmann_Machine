# AI Boltzmann Machine - Deep Learning Research Project

## 🎯 Project Overview

This project implements various energy-based neural network models for pattern recognition and digit classification. It includes implementations of **Hopfield Networks**, **Boltzmann Machines**, **Restricted Boltzmann Machines (RBMs)**, and **Deep Belief Networks (DBNs)** applied to the MNIST handwritten digit dataset.

## 🔬 What This Project Does

### Core Implementations

**1. Hopfield Networks**
- Stores and retrieves patterns using energy minimization
- Demonstrates content-addressable memory
- Handles noisy or incomplete input patterns

**2. Boltzmann Machines**
- Stochastic neural network with probabilistic updates
- Uses Gibbs sampling for inference
- Trained via Contrastive Divergence algorithm

**3. Restricted Boltzmann Machines (RBMs)**
- Unsupervised feature learning from MNIST images
- Extracts meaningful representations for classification
- Achieves 97.33% accuracy with SVM classifier

**4. Deep Belief Networks (DBNs)**
- Multi-layer architecture with layer-wise pretraining
- Hierarchical feature learning
- Fine-tuned for supervised classification

## 🚀 Key Results

- **Classification Accuracy**: 97.33% on MNIST test set
- **Reconstruction Quality**: SSIM ~0.95, PSNR ~35 dB
- **Model Architecture**: 784 input → 256 hidden units
- **Deployment**: REST API for real-time digit prediction

## 🛠️ Technologies Used

- **PyTorch**: Neural network implementation
- **scikit-learn**: SVM and Logistic Regression classifiers
- **FastAPI**: REST API deployment
- **NumPy & Matplotlib**: Data processing and visualization

## 📊 Project Structure

```
AI_Boltzmann_Machine/
├── hopfield_and_boltzmann.ipynb    # Hopfield & Boltzmann implementations
├── rbm_digit_classifier.ipynb      # RBM digit classifier with API
└── models/                          # Trained model weights
```

## 🔧 Quick Start

### Installation
```bash
pip install torch torchvision numpy matplotlib scikit-learn scikit-image fastapi uvicorn
```

### Training
```python
# Initialize and train RBM
rbm = RBM(num_visible=784, num_hidden=256)
rbm.train_rbm(train_loader, lr=0.001, epochs=50)
```

### API Usage
```bash
# Start server
uvicorn main:app --host 0.0.0.0 --port 8000

# Make prediction
curl -X POST "http://localhost:8000/predict/" -F "file=@digit.jpg"
```

## 📝 Mathematical Foundation

**Energy Function (RBM)**
```
E(v,h) = -∑ᵢ aᵢvᵢ - ∑ⱼ bⱼhⱼ - ∑ᵢⱼ vᵢWᵢⱼhⱼ
```

**Contrastive Divergence Update**
```
ΔW ∝ ⟨vhᵀ⟩data - ⟨vhᵀ⟩model
```

## 📚 References

1. Hinton, G. E. (2002). "Training Products of Experts by Minimizing Contrastive Divergence"
2. Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the Dimensionality of Data with Neural Networks"
3. Fischer, A., & Igel, C. (2012). "An Introduction to Restricted Boltzmann Machines"

## 👨💻 Author

**Jeevan Chhetri**  
Deep Learning & Computer Vision
