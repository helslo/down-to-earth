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


def weighted_mse_loss(preds: dict, targets: dict, task_weights: torch.Tensor, task_names: list):
    losses = []
    weights = []
    for i, t in enumerate(task_names):
        w = F.softplus(task_weights[i])
        mse = F.mse_loss(preds[t], targets[t])
        losses.append(w * mse)
        weights.append(w)
    return torch.stack(losses).sum() / torch.stack(weights).sum()


def ec_schedule(epoch: int, total_epochs: int) -> float:
    """Linear Exclusive Capacity schedule: EC(0) = 1.0, EC(total_epochs) = 0.0."""
    return max(0.0, 1.0 - epoch / total_epochs)


def total_loss(preds, targets, model, task_names, epoch, total_epochs, alpha_raw):
    """
    preds, targets: dict[task_name -> (B,) tensor]
    model: the FTIRNet instance (needed for its mask KL loss)
    alpha_raw: an *unconstrained* scalar tensor/nn.Parameter (Eq. 12).
        Passed through softplus here so it can only scale the MSE term
        up or down (alpha > 0), never invert or destabilize it. The
        paper only says alpha is "a trainable scale parameter" without
        specifying a constraint -- softplus is a minimal, safe choice
        consistent with how the paper already constrains the per-task
        weights (Eq. 13).
    Returns: (loss tensor, dict of logging scalars)
    """
    ec = ec_schedule(epoch, total_epochs)
    lambda_kl = 1.0 - ec

    alpha = F.softplus(alpha_raw)
    l_mse = weighted_mse_loss(preds, targets, model.task_weights, task_names)
    l_kl = model.kl_loss(ec)

    loss = alpha * l_mse + lambda_kl * l_kl
    logs = {
        "mse": l_mse.item(),
        "kl": l_kl.item(),
        "ec": ec,
        "lambda_kl": lambda_kl,
        "alpha": alpha.item(),
    }
    return loss, logs
