import torch
import math

def soft_thresholding(x, threshold):
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)

def L1Lasso(A, Y, alpha=2.812e-01, tol=1e-4, max_iter=1000):
    """
    PyTorch-based FISTA (Fast Iterative Soft Thresholding Algorithm) for GPU-accelerated Lasso.
    A: [m, n], Y: [m, batch]
    Targeting: min 1/(2m) ||Y - AX||^2 + alpha ||X||_1
    """
    m, n = A.shape
    device = A.device
    At = A.t()
    
    # Step size selection: 1 / L where L is (1/m) * sigma_max(A)^2
    with torch.no_grad():
        L = torch.linalg.matrix_norm(A, ord=2)**2 / m
        step_size = 1.0 / L
    
    X = torch.zeros((n, Y.shape[1]), device=device)
    Y_accel = X.clone()
    t = 1.0
    
    inv_m = 1.0 / m
    threshold = alpha * step_size
    
    for i in range(max_iter):
        X_old = X.clone()
        
        # Gradient of (1/2m) ||AX - Y||^2 is (1/m) A^T (AX - Y)
        grad = inv_m * (At @ (A @ Y_accel - Y))
        
        # Proximal step
        X = soft_thresholding(Y_accel - step_size * grad, threshold)
        
        # FISTA acceleration
        t_next = (1.0 + (1.0 + 4.0 * t**2)**0.5) / 2.0
        Y_accel = X + ((t - 1.0) / t_next) * (X - X_old)
        t = t_next
        
        # Convergence check
        if i > 5 and torch.norm(X - X_old) < tol * torch.norm(X):
            break
            
    return X
