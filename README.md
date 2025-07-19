# 🔢 MNIST Digit Recognition with Deep Learning

This notebook implements a comprehensive deep learning solution for handwritten digit recognition using the MNIST dataset, applying neural networks with TensorFlow/Keras to achieve high accuracy in digit classification.

---

## 📚 Table of Contents

- [💻 Installation](#-installation)
- [🎯 Project Goals](#-project-goals)
- [🧠 Methods](#-methods)
- [📈 Results](#-results)
- [🔍 Key Insights](#-key-insights)
- [🧾 Project Details](#-project-details)
- [🪪 License](#-license)

---

## 💻 Installation

- **Requirements:** Python 3.8+, Jupyter Notebook, TensorFlow 2.x

```bash
# Clone the repository
git clone https://github.com/alhussienhazem/mnist-digit-recognition.git

# Navigate to the project folder
cd mnist-digit-recognition

# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate  # on Linux/Mac
venv\Scripts\activate     # on Windows

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook
```

---

## 🎯 Project Goals

- **Objective:** Build a deep learning model to recognize handwritten digits (0-9) from the MNIST dataset
- **Challenge:** Achieve high accuracy in digit classification while handling class imbalance
- **Solution:** Implement a neural network with proper data preprocessing and class balancing
- **Features Used:** 28x28 pixel grayscale images of handwritten digits
- **Target:** Multi-class classification (10 classes: digits 0-9)

---

## 🧠 Methods

- **Models Used:**
  - 🧠 Sequential Neural Network
  - 🔢 Multi-layer Perceptron (MLP)
  - ⚖️ Class Balancing with SMOTE (Synthetic Minority Over-sampling Technique)

- **Techniques Applied:**
  - 🔍 Comprehensive data visualization and exploration
  - 📊 Class distribution analysis and imbalance handling
  - 🔧 Data normalization (0-1 scaling)
  - 🧠 Neural network architecture with multiple dense layers
  - ⚡ Adam optimizer with sparse categorical crossentropy loss
  - 📈 Early stopping and validation monitoring
  - 🎯 Confusion matrix and classification report analysis
  - 📊 Training history visualization

---

## 📈 Results

### **Model Performance**
| Metric | Score | Performance |
|--------|-------|-------------|
| 🎯 Accuracy | 98.34% | **Excellent Performance** |
| 📊 Precision | 98.0% | High Precision |
| 🔄 Recall | 98.0% | High Recall |
| ⚖️ F1-Score | 98.0% | Balanced Performance |

### **SMOTE Enhanced Model Performance**
| Metric | Score | Performance |
|--------|-------|-------------|
| 🎯 Training Accuracy | 99.83% | **Outstanding Performance** |
| 📊 Validation Accuracy | 98.35% | Excellent Performance |
| 🔄 Final Test Accuracy | 98.34% | Excellent Performance |

---

### **Model Architecture**
| Layer | Type | Units | Activation |
|-------|------|-------|------------|
| 🔢 Input | Flatten | 784 | - |
| 🧠 Hidden 1 | Dense | 1025 | ReLU |
| 🧠 Hidden 2 | Dense | 512 | ReLU |
| 🧠 Hidden 3 | Dense | 256 | ReLU |
| 🎯 Output | Dense | 10 | Softmax |

---

### **🔍 Key Insights**
- **Enhanced Accuracy**: Achieved 98.34% accuracy on test set with SMOTE, demonstrating excellent digit recognition capability
- **Advanced Class Balancing**: Successfully implemented SMOTE (Synthetic Minority Over-sampling Technique) for superior class balance
- **Improved Training**: SMOTE model achieved 99.83% training accuracy, showing outstanding learning capability
- **Data Preprocessing**: Proper normalization (0-1 scaling) and SMOTE resampling significantly improved model performance
- **Neural Network Design**: Multi-layer architecture with ReLU activations proved effective for digit classification
- **Training Efficiency**: Both models converged quickly with 10 epochs, showing good learning dynamics
- **Visualization**: Comprehensive visualization of training progress and prediction results for both approaches
- **Robust Performance**: Consistent high performance across all digit classes (0-9) with enhanced SMOTE model

---

## 🧾 Project Details

**🔧 Enhanced Data Preprocessing:**
- MNIST dataset loading and exploration
- Data shape analysis and validation
- Class distribution visualization
- SMOTE (Synthetic Minority Over-sampling Technique) for advanced class balancing
- Data normalization (0-1 scaling)

**📊 Exploratory Data Analysis:**
- Sample digit visualization
- Class distribution analysis
- Data shape and structure exploration
- Comprehensive digit sample display
- Statistical summary analysis

**🧠 Neural Network Implementation:**
- Sequential model architecture
- Multiple dense layers with ReLU activation
- Softmax output layer for multi-class classification
- Adam optimizer with sparse categorical crossentropy
- Batch training with validation monitoring

**📈 Model Evaluation:**
- Accuracy, precision, recall, and F1-score metrics
- Confusion matrix visualization
- Classification report analysis
- Training history plots (accuracy and loss)
- Prediction visualization with confidence scores

**📋 Notebook Structure:**
- Data Loading & Initial Exploration
- Data Visualization & Analysis
- Data Preprocessing & Normalization
- SMOTE Class Balancing Implementation
- Neural Network Architecture
- Model Training & Validation (Standard & SMOTE Enhanced)
- Performance Evaluation & Visualization
- Model Prediction Results & Analysis

---

## 🪪 License

This project is for educational and non-commercial use only.  
Dataset source: [TensorFlow MNIST Dataset](https://www.tensorflow.org/datasets/catalog/mnist) 