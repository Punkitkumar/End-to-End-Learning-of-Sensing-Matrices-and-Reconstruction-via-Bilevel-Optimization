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
```

### 📦 Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### ▶️ Start Training

```bash
python3 train.py
```

### 🔄 Resume Training from a Checkpoint

```bash
# Resume from the latest checkpoint (recommended after interruptions)
python3 train.py --resume checkpoints/latest.pt

# Resume from the best model
python3 train.py --resume checkpoints/best_model.pt

# Resume from a specific epoch
python3 train.py --resume checkpoints/checkpoint_epoch_10.pt
```

### 💾 Checkpointing Details

Checkpoints are saved to the `checkpoints/` directory and contain:

| Field | Description |
|---|---|
| `A_data` | Learned sensing matrix |
| `regularizer_state_dict` | Sigma, Lambda0, W parameters |
| `optimizer_state_dict` | Adam momentum buffers & LR |
| `scheduler_state_dict` | LR scheduler state |
| `best_nmse` | Best test NMSE seen so far |
| `epoch` | Epoch number |

**Saving schedule:**
- `best_model.pt` — Updated whenever a new best Test NMSE is achieved
- `latest.pt` — Saved every epoch for easy resume
- `checkpoint_epoch_X.pt` — Periodic snapshots every 5 epochs (configurable via `SAVE_EVERY` in `config.py`)

---

## 📁 Project Structure

```
.
├── config.py              # Hyperparameters and global settings
├── train.py               # Main training script (with resume support)
├── requirements.txt       # Python dependencies
├── src/
│   ├── data_loader.py     # SparseSignalsDataset and collate function
│   ├── models.py          # SmoothL1Regularizer
│   ├── solvers.py         # GPU-accelerated FISTA Lasso solver
│   ├── bilevel.py         # HOAG algorithm and inner optimization
│   └── utils.py           # NMSE, support recovery, and outer loss
├── checkpoints/           # Saved model checkpoints
├── sparse_signal_n100/    # Signal datasets (.npy files)
└── MTP.ipynb              # Original notebook (reference)
```

---

## ⚙️ Configuration

All hyperparameters are centralized in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `M` | 15 | Sensing matrix rows |
| `N` | 100 | Signal length |
| `BATCH_SIZE` | 64 | Training batch size |
| `MAX_EPOCHS` | 100 | Number of training epochs |
| `LEARNING_RATE_A` | 0.1 | LR for sensing matrix |
| `LEARNING_RATE_REG` | 0.001 | LR for regularizer params |
| `SAVE_EVERY` | 5 | Checkpoint frequency (epochs) |
