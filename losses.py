"""
Loss functions for FTIRNet (Section III-D of the paper).

L_total = alpha * L_MSE + lambda_KL * L_KL

- L_MSE: task weights are softplus'd learnable scalars, normalized by
  their sum (Eq. 13-15).
- lambda_KL = 1 - EC(t), where EC(t) linearly decays from 1 to 0 over
  training (Eq. 8).
- L_KL: KL divergence between each task's learned neuron-mask
  distribution and its EC-scheduled target distribution (Eq. 16),
  computed inside the model via `model.kl_loss(ec)`.
"""
import torch
import torch.nn.functional as F


def gaussian_nll_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor):
    var = torch.exp(log_var)
    return 0.5 * (torch.log(var) + (target - mean).pow(2) / var)


def weighted_mse_loss(preds: dict, targets: dict, task_weights: torch.Tensor, task_names: list):
    losses = []
    weights = []
    for i, t in enumerate(task_names):
        w = F.softplus(task_weights[i])
        mean, log_var = preds[t]
        mse = F.mse_loss(mean, targets[t])
        losses.append(w * mse)
        weights.append(w)
    return torch.stack(losses).sum() / torch.stack(weights).sum()


def weighted_gaussian_nll_loss(preds: dict, targets: dict, task_weights: torch.Tensor, task_names: list):
    losses = []
    weights = []
    for i, t in enumerate(task_names):
        w = F.softplus(task_weights[i])
        mean, log_var = preds[t]
        nll = gaussian_nll_loss(mean, log_var, targets[t]).mean()
        losses.append(w * nll)
        weights.append(w)
    return torch.stack(losses).sum() / torch.stack(weights).sum()


def ec_schedule(epoch: int, total_epochs: int) -> float:
    """Linear Exclusive Capacity schedule: EC(0) = 1.0, EC(total_epochs) = 0.0."""
    return max(0.0, 1.0 - epoch / total_epochs)


def total_loss(preds, targets, model, task_names, epoch, total_epochs, alpha_raw):
    """
    preds: dict[task_name -> (mean, log_var)] for each task
    targets: dict[task_name -> (B,) tensor]
    model: the FTIRNet instance (needed for its mask KL loss)
    alpha_raw: a trainable positive scale on the regression term.

    Returns: (loss tensor, dict of logging scalars)
    """
    ec = ec_schedule(epoch, total_epochs)
    lambda_kl = 1.0 - ec

    alpha = F.softplus(alpha_raw)
    l_nll = weighted_gaussian_nll_loss(preds, targets, model.task_weights, task_names)
    l_kl = model.kl_loss(ec)

    loss = alpha * l_nll + lambda_kl * l_kl
    logs = {
        "nll": l_nll.item(),
        "kl": l_kl.item(),
        "ec": ec,
        "lambda_kl": lambda_kl,
        "alpha": alpha.item(),
    }
    return loss, logs
