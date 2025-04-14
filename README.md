# 📦 End-to-End Learning of Sensing Matrices for Sparse Signal Recovery via Bilevel Optimization

This repository provides a complete pipeline for **compressed sensing** and **sparse signal recovery**, combining the power of **Lasso regression**, **learned sensing matrices**, and **bilevel optimization** (HOAG). The project is based on the M.Tech thesis work titled:

> _"End-to-End Learning of Sensing Matrices and Reconstruction via Bilevel Optimization"_  
> by **Punkit Kumar**, under the guidance of **Prof. Subhadip Mukherjee**, IIT Kharagpur.

---

## 📌 Abstract

Traditional compressed sensing uses fixed random matrices (like Gaussian), which are suboptimal when signal structures vary across datasets. This project introduces an **end-to-end bilevel learning framework** that:

- **Learns the sensing matrix A**
- **Learns smooth $\ell_1$ regularization parameters**
- **Improves both reconstruction accuracy and support recovery**

By leveraging the **HOAG algorithm**, the model differentiates through the reconstruction process to optimize the sensing matrix for any given data distribution.

---

## 🧠 Key Concepts

- **Compressed Sensing (CS)**: Reconstruct sparse signals from fewer measurements than traditional Nyquist sampling.
- **Bilevel Optimization**: Nested learning framework optimizing both sensing matrix and recovery parameters.
- **Smooth $\ell_1$ Regularization**: Ensures stable, differentiable sparsity enforcement.
- **HOAG Algorithm**: Efficient approximation-based hyperparameter optimizer.
- **Signal Reconstruction**: Lasso-based and learned approaches for high-fidelity signal recovery.

---

## 🚀 Features

- ✅ End-to-end learning of sensing matrices
- ✅ Sparse recovery using Lasso and Smooth $\ell_1$-regularized solvers
- ✅ Bilevel optimization using the HOAG algorithm
- ✅ Metrics: **Normalized Mean Squared Error (NMSE)** and **Support Recovery Ratio**
- ✅ Gaussian vs Learned matrix comparison
- ✅ PyTorch-powered implementation
- ✅ Visualizations of signal recovery and training performance

---

## 📈 Evaluation Results

**Performance at Different Compression Levels:**

| Compression (m) | NMSE (Learned) | NMSE (Gaussian) | Support (Learned) | Support (Gaussian) |
|-----------------|----------------|------------------|-------------------|---------------------|
| m = 15          | 0.646          | 0.760            | 0.337             | 0.298               |
| m = 25          | 0.238          | 0.270            | 0.605             | 0.577               |
| m = 40          | 0.002          | 0.005            | 0.980             | 0.980               |

---

## 📉 Metrics

### 🔹 Normalized Mean Squared Error (NMSE)
Measures reconstruction fidelity:

$$
\text{NMSE} = \frac{\|x_{\text{true}} - x_{\text{reconstructed}}\|^2}{\|x_{\text{true}}\|^2}
$$

### 🔹 Support Recovery
How well the non-zero positions in the sparse signal are recovered:

$$
\text{Support} = \frac{|S_{\text{true}} \cap S_{\text{recovered}}|}{|S_{\text{true}}|}
$$

---

## 🧪 How to Run

### 🔧 Installation

```bash
git clone https://github.com/Punkitkumar/End-to-End-Learning-of-Sensing-Matrices-and-Reconstruction-via-Bilevel-Optimization.git
cd End-to-End-Learning-of-Sensing-Matrices-and-Reconstruction-via-Bilevel-Optimization
!python MTP.ipynb
