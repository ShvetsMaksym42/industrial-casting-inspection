## 1. Introduction and Problem Statement

### Overview
This project addresses a key challenge in modern industrial casting: quality control through automated visual inspection. 
Traditional manual inspection suffers from:
* **Human Factor:** Fatigue, subjectivity, and inconsistent defect detection.
* **Scalability:** On high-speed production lines, humans physically cannot inspect every single part.
* **Cost:** Maintaining a large staff of inspectors is expensive for the enterprise.

This project demonstrates the use of **Computer Vision (CV)** to automate the classification of casting parts into **"OK"** (defective-free) and **"Defective"** categories.

### The Challenge (Accuracy vs. Resources)
Automation in this field faces a classic engineering dilemma:
1. **Complex Architectures (e.g., ResNet18):** Provide near-perfect accuracy but require powerful GPUs, which are expensive to deploy on the factory floor.
2. **Lightweight Architectures (Custom CNN):** Run fast even on cheap CPUs/edge devices but might struggle with subtle or complex defect patterns.

**Goal:** Develop and compare two solutions — a highly efficient **Custom CNN** and a robust **Pre-trained ResNet18** — to find the optimal balance for industrial deployment.

---

## 2. Dataset

The models were trained and evaluated using the **Casting Product Image Data for Quality Inspection** dataset.

* **Source:** [Casting Product Dataset on Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/data)
* **Content:** 1,300 images of top-view of submersible pump impeller.
* **Classes:**

    * **Defective** (Casting defects like blow holes, pinholes, burr, shrinkage defects, mould material defects, pouring metal defects, metallurgical defects, etc.).

    * **OK** (Clean, defect-free products)
* **Environment:** Captured under stable industrial lighting, but includes challenges such as varying orientations and different types of casting flaws.

* **Dataset constraints:** Small and imbalanced, which makes model training challenging.
* **Solution:** To prevent overfitting and improve generalization, **extensive data augmentation** was applied, including:

    * Random rotations and horizontal/vertical flips.
    * Brightness and contrast adjustments.
    * Perspective transforms to simulate camera positioning variances.
  
Note: Small and imbalanced datasets are very common in real-world industrial settings, making these techniques directly relevant for practical deployment.  

---

## 3. Development Pipeline

The project implements a full end-to-end pipeline for automated defect detection, covering everything from raw data processing to deep model interpretability.

### 1. Custom Dataset & Data Engineering
A robust `Dataset` class was engineered in PyTorch to handle the specific needs of industrial imagery:
* **Dynamic Loading:** Efficient memory management for high-resolution casting images.
* **Extensive Augmentation:** To combat the small dataset size, I implemented a pipeline including random rotations, flips, and color jittering to simulate varying factory floor conditions.
* **Balanced Sampling:** Strategic shuffling and batching to ensure the model doesn't develop a bias toward "OK" products.

### 2. Model Architectures
Two distinct philosophies were compared to find the industrial "sweet spot":
* **ResNet18 (The Benchmark):** A pre-trained industry-standard architecture. While it provides a high-accuracy baseline, its size (**64.5 MB**) presents challenges for edge-device deployment.
* **Custom CNN (The Optimized Solution):** A bespoke architecture inspired by ResNet principles.
    * **Features:** Utilizes **Skip Connections** and **Residual Blocks** to ensure smooth gradient flow.
    * **Efficiency:** Optimized to a compact **4 MB**, making it 16x smaller than ResNet18 while maintaining competitive performance.
    * **Design:** Specifically tuned to capture subtle casting flaws without the overhead of massive parameters.

### 3. Training & Regularization
* **Optimization:** Used Binary Cross-Entropy loss incorporating a **`pos_weight` parameter** to handle class imbalance. I leveraged the **Adam optimizer** for its efficient adaptive learning rate capabilities, ensuring stable convergence.
* **Regularization & Generalization:** Given the relatively small dataset size, I applied a multi-layered regularization strategy to prevent overfitting:
    * **Dropout Layers:** To de-activate neurons randomly during training, forcing the network to learn robust, non-redundant features.
    * **Weight Decay (L2 Regularization):** To penalize large weights and maintain a simpler, more generalizable model.
    * **Data Augmentation:** To synthetically expand the training variety, ensuring the models don't just "memorize" the limited training set.

### 4. Evaluation & Interpretability
* **Performance Analysis:** Evaluation was conducted using **Confusion Matrices** to look beyond simple accuracy. This allowed for a detailed breakdown of classification errors, specifically identifying **False Negatives**.
* **Explainable AI (Grad-CAM):** Integrated Heatmaps to verify that the model is actually looking at **defects** (cracks, holes) rather than background noise or lighting artifacts.
* **Trade-off Analysis:** A final comparison of model size vs. inference speed vs. accuracy to determine the best candidate for production.
