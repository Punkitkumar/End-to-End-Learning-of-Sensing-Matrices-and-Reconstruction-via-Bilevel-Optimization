import torch

def sup_xxhat(x_true, x_hat, device='cpu'):
    non_zero_mask_true = x_true != 0 
    non_zero_mask_hat = x_hat != 0  
    top_10_indices_true = torch.argsort(torch.abs(x_true), dim=0, descending=True)[:10, :]
    top_10_indices_hat = torch.argsort(torch.abs(x_hat), dim=0, descending=True)[:10, :]
    
    top_10_mask_true = torch.zeros_like(x_true, dtype=torch.bool, device=device)
    top_10_mask_hat = torch.zeros_like(x_hat, dtype=torch.bool, device=device)
    top_10_mask_true.scatter_(0, top_10_indices_true, True)
    top_10_mask_hat.scatter_(0, top_10_indices_hat, True)
    
    support_true = (non_zero_mask_true & top_10_mask_true) 
    support_hat = (non_zero_mask_hat & top_10_mask_hat) 
    
    intersection = (support_true & support_hat).sum(dim=0).float() 
    union = (support_true).sum(dim=0).float()
    
    # Avoid division by zero
    support_ratios = intersection / (union + 1e-8)
    return support_ratios.sum().item()

def nmse_xxhat(x_true, x_hat, device='cpu'):
    numerator = torch.sum((x_true - x_hat)**2 , dim=0) 
    denominator = torch.sum(x_true**2 , dim=0)          
    nmse_per_sample = numerator / (denominator + 1e-8) 
    return torch.sum(nmse_per_sample).item()
    
def signal_recovery_eval(x_hat, x_true, device):
    sum_nmse = nmse_xxhat(x_true=x_true, x_hat=x_hat, device=device)
    sum_support = sup_xxhat(x_true=x_true, x_hat=x_hat, device=device)
    return sum_nmse, sum_support

def outer_loss_fn(pred_x, x_true):
    import torch.nn as nn
    criterion = nn.MSELoss(reduction='none')
    mse_loss_per_feature = criterion(pred_x, x_true)  # Shape: [n, batch]
    mse_loss_per_sample = mse_loss_per_feature.sum(dim=0)  # Shape: [batch]
    return mse_loss_per_sample.mean()
