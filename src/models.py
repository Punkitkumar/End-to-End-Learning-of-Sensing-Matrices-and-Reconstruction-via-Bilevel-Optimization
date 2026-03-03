import torch
import torch.nn as nn

class SmoothL1Regularizer(nn.Module):
    def __init__(self, n, init_sigma=-7.0, init_lambda0=0.0, device='cpu'):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(init_sigma, dtype=torch.float32, device=device))
        self.lambda0 = nn.Parameter(torch.tensor(init_lambda0, dtype=torch.float32, device=device))
        self.W = nn.Parameter(torch.eye(n, dtype=torch.float32, device=device))

    def forward(self, x_hat):
        """Computes the SmoothL1 regularization term"""
        transform_xhat = self.W @ x_hat
        regularization = torch.sqrt(transform_xhat**2 + torch.exp(self.sigma))
        return torch.exp(self.lambda0) * torch.sum(regularization, dim=0)
