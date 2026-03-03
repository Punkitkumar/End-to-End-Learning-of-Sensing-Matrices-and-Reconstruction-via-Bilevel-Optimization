import torch

# --- Signal & Measurement Dimensions ---
M = 15             # Sensing matrix rows (measurements)
N = 100            # Signal length (features)

# --- Training Hyperparameters ---
BATCH_SIZE = 64
MAX_EPOCHS = 100
LEARNING_RATE_A = 0.1
LEARNING_RATE_REG = 0.001
WEIGHT_DECAY = 1e-2

# --- Dataset Settings ---
TRAIN_DATA_PATH = 'sparse_signal_n100/sparse_signals_dataset_10000.npy'
TEST_DATA_PATH = 'sparse_signal_n100/sparse_signals_dataset_1000.npy'
TRAIN_SAMPLES = 4500
TEST_SAMPLES = 1000

# --- Inner Optimization (LBFGS) Parameters ---
LBFGS_LR = 0.1
LBFGS_TOL_GRAD = 1e-8
LBFGS_TOL_CHG = 1e-10
LBFGS_MAX_ITER = 500

# --- HOAG & Solver Parameters ---
HOAG_EPSILON = 1e-5
HOAG_K = 1.0       # Lipschitz constant approximation factor
LASSO_TOL = 1e-4
LASSO_MAX_ITER = 1000
LASSO_ALPHA_EVAL = 2.812e-01 # Default alpha for baseline Lasso comparison

# --- Checkpointing & Logging ---
CHECKPOINT_DIR = 'checkpoints'
SAVE_EVERY = 5     # Save full checkpoint every 5 epochs
LOG_INTERVAL = 10  # Future use for detailed logging frequency

# --- Device Selection ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
