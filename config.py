import torch

# Training parameters
M = 15  # Sensing matrix rows
N = 100 # Signal length
BATCH_SIZE = 64
MAX_EPOCHS = 100
LEARNING_RATE_A = 0.1
LEARNING_RATE_REG = 0.001
WEIGHT_DECAY = 1e-2

# File paths
TRAIN_DATA_PATH = 'sparse_signal_n100/sparse_signals_dataset_10000.npy'
TEST_DATA_PATH = 'sparse_signal_n100/sparse_signals_dataset_1000.npy'
TRAIN_SAMPLES = 4500
TEST_SAMPLES = 1000

# Optimization parameters
HOAG_EPSILON = 1e-5
LASSO_TOL = 1e-4
LASSO_MAX_ITER = 1000

# Checkpointing
CHECKPOINT_DIR = 'checkpoints'
SAVE_EVERY = 5 # Save every 5 epochs

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
