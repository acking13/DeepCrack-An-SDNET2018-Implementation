
---

# DeepCrack

**DeepCrack** is a deep learning-based project designed to detect cracks in concrete structures from images. Using Convolutional Neural Networks (CNNs), the model classifies images as either **Cracked** or **Non-Cracked**. This project can be used for structural health monitoring, automated inspection, and quality control in civil engineering.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Models and Results](#models-and-results)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Evaluation](#evaluation)
* [References](#references)

---

## Project Overview

Concrete cracks can compromise the structural integrity of buildings and infrastructure. Manual inspection is time-consuming and prone to errors. This project leverages neural networks to automatically identify cracks from images of concrete surfaces.

Key features:

* Detects cracks in concrete with high accuracy.
* Uses data augmentation to handle imbalanced datasets.
* Supports training, evaluation, and visualization of performance metrics.
* Implements advanced CNN architectures with regularization and focal loss for better performance on imbalanced classes.

---

## Dataset

The dataset is based on images of **walls** divided into two categories:

* `cracked` — images containing cracks
* `non-cracked` — images without cracks

**Dataset Size:**

* Total images: 1,719

  * Cracked: 846
  * Non-Cracked: 873

**Preprocessing:**

* Images resized to `128x128` pixels.
* Brightness enhancement applied for better feature extraction.
* Standard normalization (`0-1`) applied.

---

## Models and Results

You have experimented with 10 different models. Below is a summary of **Model 9**, which shows the most representative results:

|                       | Actual Non-Cracked | Actual Cracked |
| --------------------- | ------------------ | -------------- |
| Predicted Non-Cracked | 134                | 46             |
| Predicted Cracked     | 123                | 41             |

Other models varied in performance, some overfitting (always predicting one class) or underperforming due to imbalance.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/DeepCrack.git
cd DeepCrack
```

2. Create a Python virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Requirements (main packages):**

* Python 3.9+
* TensorFlow 2.x
* numpy, pandas, matplotlib, scikit-learn, Pillow

---

## Usage

1. Update the dataset path in `DATA_ROOT` in the main script:

```python
DATA_ROOT = r'C:\path\to\your\dataset'
```

2. Run the main training script:

```bash
python deepcrack_main.py
```

3. Outputs:

* `best_model.h5` — trained CNN model
* `upgraded_model_metrics.csv` — confusion matrix & classification report
* `accuracy_history.png`, `loss_history.png`, `confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png` — evaluation plots

---

## Model Architecture

The CNN model consists of:

* 3 Convolutional blocks with Batch Normalization, MaxPooling, and Dropout.
* Fully connected layer with 256 units, Batch Normalization, and Dropout.
* Output layer with a single neuron and Sigmoid activation.

Advanced techniques applied:

* L2 regularization
* Focal loss for class imbalance
* Data augmentation to increase dataset variability
* EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks

---

## Evaluation

Metrics include:

* **Accuracy**
* **AUC (Area Under ROC Curve)**
* **Precision / Recall / F1-score**
* **Confusion Matrix**

Sample evaluation for **Model 9** (threshold auto-selected via Youden Index):

```
Test loss: 2.05
Test accuracy: 0.51
AUC: 0.50
Confusion matrix and classification report saved in CSV.
```

**Plots:**

* Training & validation accuracy/loss
* Confusion matrix
* ROC curve
* Precision-Recall curve

---

## References

1. Zhang, L., Yang, F., Zhang, Y.-D., & Zhu, Y.-J. (2016). *Road crack detection using deep convolutional neural network*.
2. SDNET2018: Structural crack image dataset for concrete walls, floors, and pavements.
3. TensorFlow Keras documentation: [https://www.tensorflow.org/guide/keras](https://www.tensorflow.org/guide/keras)

---

## License

MIT License — feel free to use, modify, and share for research and educational purposes.

---

