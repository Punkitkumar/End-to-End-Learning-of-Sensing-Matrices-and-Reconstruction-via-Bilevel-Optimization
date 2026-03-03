import torch
import torch.autograd as autograd
import torch.optim as optim

def compute_Hv(loss, p, u, v, flag="both"): 
    grad_u = autograd.grad(loss, u, create_graph=True)[0]
    
    if flag == "grad":
        return grad_u
    
    elif flag == "both":
        dell_u_times_p = torch.sum(grad_u * p)
        Hv = autograd.grad(dell_u_times_p, v, retain_graph=True)[0]
        return grad_u, Hv
    
    elif flag == "hess":
        dell_u_times_p = torch.sum(grad_u * p)
        Hv = autograd.grad(dell_u_times_p, v, retain_graph=True)[0]
        return Hv

def create_H_function(loss, u, v):
    def H(p):
        return compute_Hv(loss, p, u, v, flag="hess")
    return H

def inner_loss(Y, A, x_hat, regularizer):
    """Computes the total loss with integrated regularizer"""
    residual = Y - A @ x_hat
    loss1 = torch.sum(residual**2, dim=0)
    regularizer_term = regularizer(x_hat)
    return (loss1 + regularizer_term).mean()

def inner_optimization(x_hat, Y, A, regularizer, lr=0.1, tol_grad=1e-8, tol_chg=1e-10, max_iter=500):
    recons_x = x_hat.clone().detach().requires_grad_(True)
    optimizer = optim.LBFGS([recons_x], 
                      lr=lr,
                      line_search_fn='strong_wolfe',  
                      tolerance_grad=tol_grad,  
                      tolerance_change=tol_chg)

    def closure():
        optimizer.zero_grad()
        loss = inner_loss(Y=Y, A=A, x_hat=recons_x, regularizer=regularizer)
        loss.backward(retain_graph=True)
        return loss
    
    optimizer.step(closure)
    return recons_x

def hoag_algorithm(x_hat, x_true, A, regularizer, outer_loss_fn, device, k=1.0, epsilon=1e-8, 
                   lbfgs_lr=0.1, lbfgs_tol_grad=1e-8, lbfgs_tol_chg=1e-10):
    """
    Optimized HOAG using direct linear solve for the inverse Hessian-vector product.
    """
    Y = (A @ x_true).to(device)
    batch_size = x_true.shape[1]
    
    # 1. Inner Optimization
    recons_x = inner_optimization(x_hat=x_hat, Y=Y, A=A, regularizer=regularizer, 
                                  lr=lbfgs_lr, tol_grad=lbfgs_tol_grad, tol_chg=lbfgs_tol_chg)
    x_hat_final = recons_x.detach().requires_grad_(True)
    
    # 2. Compute Outer Gradient wrt x
    outer_loss = outer_loss_fn(x_hat_final, x_true)
    grad_g_x = torch.autograd.grad(outer_loss, x_hat_final, create_graph=True)[0] # [n, batch]
    
    # 3. Solve (Hxx) q = grad_g_x per sample in batch
    with torch.no_grad():
        W = regularizer.W
        lam = torch.exp(regularizer.lambda0)
        eps = torch.exp(regularizer.sigma)
        Wx = W @ x_hat_final
        d = eps / (Wx**2 + eps)**1.5 # [n, batch]
        
        A2 = 2 * (A.t() @ A) # [n, n]
        W_t = W.t()
        
        # d.T: [batch, n]
        batch_H_reg = W_t.unsqueeze(0) @ (d.T.unsqueeze(2) * W.unsqueeze(0))
        batch_H = (A2.unsqueeze(0) + lam * batch_H_reg) / batch_size
        
        # Solve H q = g (transpose g for batch solve)
        q_k_batch = torch.linalg.solve(batch_H, grad_g_x.T.unsqueeze(2)).squeeze(2) # [batch, n]
        q_k = q_k_batch.T # [n, batch]
    
    # 4. Compute Hypergradients wrt A, sigma, lambda0, W
    params = [A] + list(regularizer.parameters())
    gradients = []
    
    new_inner_loss = inner_loss(Y=Y, A=A, x_hat=x_hat_final, regularizer=regularizer)
    for param in params:
        Hv = compute_Hv(new_inner_loss, q_k, x_hat_final, param, flag="hess")
        gradients.append(-Hv)
    
    L = k * torch.linalg.matrix_norm(grad_g_x).mean().item()
    
    return gradients + [L, outer_loss.item(), x_hat_final]
