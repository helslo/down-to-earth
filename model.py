"""
FTIRNet: a from-scratch PyTorch reimplementation of the architecture in
"Science-Informed Multitask Transformer for Soil Property Prediction from
FTIR Spectroscopy" (Bachinin et al., IEEE eScience 2025).

This module implements:
  - SharedEncoder      : the shared transformer encoder (Section III-A)
  - TaskMask            : the learnable per-neuron/per-task mask and its
                           Exclusive-Capacity (EC) target distribution
                           (Section III-B.2, Eq. 4-9)
  - TaskHead             : a task-specific transformer head with
                           attention pooling (Section III-B.1)
  - FusionGate            : the science-informed gamma-gated fusion layer
                           between an influencing task and a listener
                           task (Section III-C, Eq. 10-11)
  - FTIRNet               : wires all of the above together (the "ELMA
                           block" in the paper is the combination of
                           TaskMask + FusionGate + TaskHead for one task)

Notes on faithfulness to the paper (read before you trust numbers):
  - The paper's 160 input features are themselves *not* raw wavelengths.
    They are the latent output of a separate autoencoder (ref [16] in the
    paper) trained on the concatenation of three preprocessed spectral
    variants (raw absorbance, SW-transform, first derivative), each
    reduced with PCA. That autoencoder is NOT described in enough detail
    to reproduce exactly, and it is NOT implemented here. You either need
    to (a) train your own compression step of similar spirit, or
    (b) start simpler and feed the shared encoder raw/derivative spectra
    directly (see README for a suggested simplification).
  - The paper's Eq. 8 defines pi as a per-neuron mix between an exclusive
    initial target and a uniform 0.5 target. The exact initial exclusive
    assignment (which neuron "belongs" to which task at t=0) is not
    specified beyond "each row is set so exactly one owner task's entry
    is 1-eps". I implement this as a round-robin assignment across tasks,
    which is a reasonable, simple choice consistent with the description.
  - alpha in the total loss is described as "a trainable scale parameter"
    with no further detail; I implement it as a single learnable scalar.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al.), applied
    over the 160 "positions" of the compressed spectral representation."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SharedEncoder(nn.Module):
    """Shared transformer encoder (Section III-A).

    Input: (batch, in_features) -- the 160-dim compressed spectral
    features (or raw/derivative spectra if you skip the autoencoder,
    see README).

    Each scalar feature is projected through a linear "embedding" layer
    to d_model, refined with batch normalization, given a positional
    encoding, then passed through `num_layers` standard transformer
    encoder layers shared across every downstream task.
    """

    def __init__(
        self,
        in_features: int = 160,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=in_features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)
        x = x.unsqueeze(-1)  # (B, L, 1)
        x = self.embed(x)  # (B, L, d_model)
        x = x.transpose(1, 2)  # (B, d_model, L) for BatchNorm1d
        x = self.bn(x)
        x = x.transpose(1, 2)  # (B, L, d_model)
        x = self.pos_enc(x)
        return self.encoder(x)  # (B, L, d_model)


class TaskMask(nn.Module):
    """Learnable per-neuron, per-task soft mask with a progressive
    Exclusive-Capacity (EC) target distribution (Eq. 4-9).

    q[i, j] = sigmoid(phi[i, j])   -- learned "how much neuron i belongs
                                       to task j" probability
    p[i, j] = target distribution, blended from a near-exclusive initial
              assignment towards a uniform 0.62-probability target as
              training progresses (EC(t): 1 -> 0).
    """

    def __init__(self, d_model: int, task_names: list, eps: float = 0.05):
        super().__init__()
        self.task_names = task_names
        n_tasks = len(task_names)
        self.phi = nn.Parameter(torch.zeros(d_model, n_tasks))

        # Round-robin exclusive initial ownership: neuron i "belongs" to
        # task (i mod n_tasks) at the start of training.
        pi_init = torch.full((d_model, n_tasks), eps)
        owners = torch.arange(d_model) % n_tasks
        pi_init[torch.arange(d_model), owners] = 1.0 - eps
        self.register_buffer("pi_init", pi_init)

    def q(self) -> torch.Tensor:
        return torch.sigmoid(self.phi)  # (d_model, n_tasks)

    def target_p(self, ec: float) -> torch.Tensor:
        # Eq. 8, applied directly to probabilities (0.5 logit ~ 0.62 prob
        # as noted in the paper).
        return ec * self.pi_init + (1.0 - ec) * 0.62

    def masked(self, z: torch.Tensor, task_idx: int) -> torch.Tensor:
        # z: (B, L, d_model) -> element-wise scale by the task's neuron mask
        q_task = self.q()[:, task_idx]  # (d_model,)
        return z * q_task.view(1, 1, -1)

    def kl_loss(self, ec: float, eps: float = 1e-6) -> torch.Tensor:
        q = self.q().clamp(eps, 1 - eps)
        p = self.target_p(ec).clamp(eps, 1 - eps)
        kl = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
        return kl.sum()


class AttentionPool(nn.Module):
    """Adaptive weighted pooling over the sequence positions, producing a
    single context vector per sample."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, L, d_model)
        weights = torch.softmax(self.score(z), dim=1)  # (B, L, 1)
        return (weights * z).sum(dim=1)  # (B, d_model)


class TaskHead(nn.Module):
    """Task-specific transformer head: transformer layers -> attention
    pooling -> linear projection to a scalar prediction."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionPool(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.encoder(z)
        pooled = self.pool(z)
        return self.out(pooled).squeeze(-1)


class FusionGate(nn.Module):
    """One directed science-informed edge: influencing task -> listener
    task (Eq. 10-11). Concatenates the listener's masked features with a
    gamma-scaled copy of the influencing task's masked features, then
    projects back to d_model with a linear + ReLU fusion layer.

    If `fixed_gamma` is given, gamma is that constant (not learned) --
    this reproduces the paper's gamma=1.0 ablation (Table III, column e).
    Otherwise gamma is a learned scalar gate initialized at sigmoid(0)=0.5,
    matching the dynamic-gamma configuration (Table III, column f).
    """

    def __init__(self, d_model: int, fixed_gamma: float = None):
        super().__init__()
        self.fixed_gamma = fixed_gamma
        if fixed_gamma is None:
            self.gamma_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5
        self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU())

    def gamma(self) -> torch.Tensor:
        if self.fixed_gamma is not None:
            return torch.tensor(self.fixed_gamma)
        return torch.sigmoid(self.gamma_logit)

    def forward(self, listener_seq: torch.Tensor, influencer_seq: torch.Tensor) -> torch.Tensor:
        gated_influencer = self.gamma() * influencer_seq
        fused = torch.cat([listener_seq, gated_influencer], dim=-1)
        return self.fuse(fused)


class FTIRNet(nn.Module):
    """Full model: shared encoder + per-task ELMA blocks (mask -> optional
    science-informed fusion -> task-specific transformer head)."""

    def __init__(
        self,
        task_names: list,
        in_features: int = 160,
        d_model: int = 256,
        nhead: int = 8,
        shared_layers: int = 3,
        task_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        fusion_edges: dict = None,
        fixed_gamma: float = None,
    ):
        """
        task_names: ordered list of property names, e.g. ['OC', 'N', 'pH'].
        fusion_edges: dict listener -> [influencer, ...], e.g.
            {'N': ['OC', 'pH']} encodes the paper's best-performing
            "OC, pH -> N" configuration (Table II, column d).
            Avoid cycles between any pair of tasks (paper's stated
            constraint).
        fixed_gamma: if given, gamma is fixed to this value instead of
            learned (paper's ablation (e) uses fixed gamma=1.0).
        """
        super().__init__()
        self.task_names = task_names
        self.fusion_edges = fusion_edges or {}
        for listener, influencers in self.fusion_edges.items():
            assert listener in task_names
            for inf in influencers:
                assert inf in task_names
                # basic cycle guard: influencer must not itself list `listener`
                assert listener not in self.fusion_edges.get(inf, []), (
                    f"Cycle detected between '{inf}' and '{listener}'"
                )

        self.shared_encoder = SharedEncoder(
            in_features, d_model, nhead, dim_feedforward, shared_layers, dropout
        )
        self.mask = TaskMask(d_model, task_names)
        self.heads = nn.ModuleDict(
            {
                t: TaskHead(d_model, nhead, dim_feedforward, task_layers, dropout)
                for t in task_names
            }
        )

        self.gates = nn.ModuleDict()
        for listener, influencers in self.fusion_edges.items():
            for inf in influencers:
                self.gates[f"{inf}->{listener}"] = FusionGate(
                    d_model, fixed_gamma=fixed_gamma
                )

        # Learnable per-task loss weights (Eq. 13-15), passed through
        # softplus at loss-computation time.
        self.task_weights = nn.Parameter(torch.zeros(len(task_names)))

    def forward(self, x: torch.Tensor, ec: float = 1.0) -> dict:
        z_shared = self.shared_encoder(x)  # (B, L, d_model)

        masked = {
            t: self.mask.masked(z_shared, i) for i, t in enumerate(self.task_names)
        }

        preds = {}
        for t in self.task_names:
            seq = masked[t]
            if t in self.fusion_edges:
                for inf in self.fusion_edges[t]:
                    gate = self.gates[f"{inf}->{t}"]
                    seq = gate(seq, masked[inf])
            preds[t] = self.heads[t](seq)
        return preds

    def kl_loss(self, ec: float) -> torch.Tensor:
        return self.mask.kl_loss(ec)
