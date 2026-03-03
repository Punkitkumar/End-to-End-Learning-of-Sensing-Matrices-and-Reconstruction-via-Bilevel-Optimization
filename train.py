import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
import os
import argparse

from config import *
from src.data_loader import SparseSignalsDataset, collate_signals
from src.models import SmoothL1Regularizer
from src.bilevel import hoag_algorithm, inner_optimization
from src.utils import signal_recovery_eval, outer_loss_fn


def save_checkpoint(epoch, A, regularizer, optimizer, scheduler, best_nmse, filename):
    checkpoint = {
        'epoch': epoch,
        'A_data': A.data.cpu(),
        'regularizer_state_dict': regularizer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_nmse': best_nmse
    }
    torch.save(checkpoint, filename)
    print(f"  Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, A, regularizer, optimizer, scheduler, device):
    """
    Load a checkpoint and restore all training state.
    Returns the starting epoch and best_nmse from the checkpoint.
    """
    print(f"\n--- Resuming from checkpoint: {checkpoint_path} ---")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore sensing matrix A
    A.data = checkpoint['A_data'].to(device)

    # Restore regularizer (sigma, lambda0, W)
    regularizer.load_state_dict(checkpoint['regularizer_state_dict'])

    # Restore optimizer and scheduler
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch']
    best_nmse = checkpoint.get('best_nmse', checkpoint.get('nmse', float('inf')))

    print(f"  Restored epoch: {start_epoch}")
    print(f"  Best NMSE so far: {best_nmse:.6e}")
    print(f"  Sigma: {regularizer.sigma.item():.4f} | Lambda0: {regularizer.lambda0.item():.4f}")
    print(f"--- Resuming training from epoch {start_epoch + 1} ---\n")

    return start_epoch, best_nmse


def train(resume_path=None):
    # 1. Initialize Datasets and Loaders
    train_dataset = SparseSignalsDataset(TRAIN_DATA_PATH, samples_number=TRAIN_SAMPLES)
    test_dataset = SparseSignalsDataset(TEST_DATA_PATH, samples_number=TEST_SAMPLES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, collate_fn=collate_signals)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, collate_fn=collate_signals)

    print(f"Training Dataset Size: {len(train_dataset)}")
    print(f"Testing Dataset Size: {len(test_dataset)}")
    print(f"Using device: {DEVICE}")

    # 2. Initialize Parameters
    A = nn.Parameter(torch.randn(M, N, dtype=torch.float32, device=DEVICE))
    regularizer = SmoothL1Regularizer(n=N, device=DEVICE).to(DEVICE)

    # 3. Initialize Optimizer and Scheduler
    optimizer = optim.Adam([
        {'params': [A], 'lr': LEARNING_RATE_A},
        {'params': regularizer.parameters(), 'lr': LEARNING_RATE_REG}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 4. Resume from checkpoint if specified
    start_epoch = 0
    best_nmse = float('inf')

    if resume_path:
        if os.path.isfile(resume_path):
            start_epoch, best_nmse = load_checkpoint(
                resume_path, A, regularizer, optimizer, scheduler, DEVICE
            )
        else:
            print(f"WARNING: Checkpoint file not found at '{resume_path}'. Starting from scratch.")

    # 5. Storage for warm start
    previous_estimates = torch.zeros((N, len(train_dataset)), device=DEVICE)

    # 6. Training Loop
    total_tr_time = 0
    train_nmse_history = []

    for epoch in range(start_epoch, MAX_EPOCHS):
        epoch_start_time = time.time()

        running_nmse_loss = 0.0
        total_train_samples = 0
        total_support_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS} Training", leave=False)
        for batch_idx, (x_true, indices) in enumerate(train_loader_tqdm):
            n_samples = x_true.shape[1]
            total_train_samples += n_samples
            x_true = x_true.to(DEVICE)

            initial_estimate = previous_estimates[:, indices]

            # HOAG Optimization Step
            grads = hoag_algorithm(
                x_hat=initial_estimate,
                x_true=x_true,
                A=A,
                regularizer=regularizer,
                outer_loss_fn=outer_loss_fn,
                device=DEVICE,
                epsilon=HOAG_EPSILON
            )

            grad_A, grad_sigma, grad_lambda0, grad_W, L, loss, x_hat = grads

            # Update stored estimate
            previous_estimates[:, indices] = x_hat.detach()

            # Update parameters
            optimizer.zero_grad()
            A.grad = grad_A
            regularizer.sigma.grad = grad_sigma
            regularizer.lambda0.grad = grad_lambda0
            optimizer.step()

            # Evaluate performance
            nmse_loss_batch, support_loss_batch = signal_recovery_eval(x_hat, x_true, DEVICE)
            running_nmse_loss += nmse_loss_batch
            total_support_loss += support_loss_batch

            train_loader_tqdm.set_postfix({
                'Outer Loss': f"{loss:.4e}",
                'Avg NMSE': f"{running_nmse_loss/total_train_samples:.4e}"
            })

        avg_train_nmse = running_nmse_loss / total_train_samples
        train_nmse_history.append(avg_train_nmse)
        scheduler.step(avg_train_nmse)

        epoch_elapsed = time.time() - epoch_start_time
        total_tr_time += epoch_elapsed

        # Testing phase
        with torch.no_grad():
            total_test_samples = 0
            total_test_nmse = 0.0
            total_test_support = 0.0

            for x_true_test, _ in test_loader:
                x_true_test = x_true_test.to(DEVICE)
                n_samples_test = x_true_test.shape[1]
                total_test_samples += n_samples_test

                Y_test = A @ x_true_test
                initial_est_test = torch.zeros_like(x_true_test)

                x_hat_test = inner_optimization(initial_est_test, Y_test, A, regularizer, tol=HOAG_EPSILON)

                nmse_test, support_test = signal_recovery_eval(x_hat_test, x_true_test, DEVICE)
                total_test_nmse += nmse_test
                total_test_support += support_test

            avg_test_nmse = total_test_nmse / total_test_samples
            avg_test_support = total_test_support / total_test_samples

        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS} Summary:")
        print(f"  Time: {epoch_elapsed:.2f}s | Train NMSE: {avg_train_nmse:.6e}")
        print(f"  Test NMSE: {avg_test_nmse:.6e} | Test Support: {avg_test_support:.4f}")

        # 7. Checkpointing — periodic save
        if (epoch + 1) % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(epoch + 1, A, regularizer, optimizer, scheduler, best_nmse, checkpoint_path)

        # Save best model
        if avg_test_nmse < best_nmse:
            best_nmse = avg_test_nmse
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            save_checkpoint(epoch + 1, A, regularizer, optimizer, scheduler, best_nmse, best_path)
            print(f"  *** New best model (NMSE: {best_nmse:.6e}) ***")

        # Always save latest checkpoint for easy resume
        latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        save_checkpoint(epoch + 1, A, regularizer, optimizer, scheduler, best_nmse, latest_path)

    print(f"\nTotal training time: {total_tr_time:.2f}s")
    print(f"Best Test NMSE achieved: {best_nmse:.6e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sensing Matrix via Bilevel Optimization")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g. checkpoints/best_model.pt or checkpoints/latest.pt)')
    args = parser.parse_args()

    train(resume_path=args.resume)
