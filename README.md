## 1. Introduction and Problem Statement

### Overview
In modern industrial casting processes, quality control is a critical stage. Traditional manual visual inspection by humans has several significant drawbacks:
* **Human Factor:** Fatigue, subjectivity, and inconsistent defect detection.
* **Scalability:** On high-speed production lines, humans physically cannot inspect every single part.
* **Cost:** Maintaining a large staff of inspectors is expensive for the enterprise.

This project demonstrates the use of **Computer Vision (CV)** to automate the classification of casting parts into "OK" (defective-free) and "Defective" categories.

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

    -Defective (Casting defects like blow holes, pinholes, burr, shrinkage defects, mould material defects, pouring metal defects, metallurgical defects, etc.).

    -OK (Clean, defect-free products)
* **Environment:** Captured under stable industrial lighting, but includes challenges such as varying orientations and different types of casting flaws.

  
