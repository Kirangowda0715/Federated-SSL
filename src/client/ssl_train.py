"""
src/client/ssl_train.py
-----------------------
Local SSL training loop for each hospital using a Masked Autoencoder (MAE).

Key design decisions:
  - Only encoder weights are returned to the server (decoder stays local)
  - FedProx proximal term added to client loss when global_weights provided
  - AdamW optimizer with cosine annealing LR schedule
"""

import copy
import math
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def ssl_local_train(
    hospital_id: int,
    model: nn.Module,
    dataloader: DataLoader,
    config,
    global_weights: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train the MAE locally on hospital data for one federated round.

    Args:
        hospital_id    : Hospital identifier (1..5) — used for logging
        model          : MaskedAutoencoder instance (encoder + decoder)
        dataloader     : DataLoader for this hospital's NIH shard
        config         : Loaded config (SimpleNamespace)
        global_weights : Encoder state_dict from the global server model.
                         If provided, FedProx proximal regularization is applied.
                         If None, standard MAE loss only (FedAvg mode).
        device         : torch.device; defaults to CUDA if available

    Returns:
        Dict with:
            'encoder_weights' : Encoder state_dict (to send to server)
            'num_samples'     : Number of training samples seen
            'epoch_losses'    : List of mean loss per epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    # ── Optimizer & scheduler ───────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.ssl.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )
    num_epochs = config.ssl.epochs_per_round
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ── FedProx: store reference to global weights on device ─────────────
    is_fedprox = (global_weights is not None) and (config.federated.aggregation == "fedprox")
    mu = getattr(config.federated, "fedprox_mu", 0.01)
    global_params_flat = None

    if is_fedprox and global_weights is not None:
        # Flatten all global encoder parameters into a single vector for proximal term
        global_params_flat = _flatten_weights(global_weights, device)

    # ── Training loop ────────────────────────────────────────────────────
    epoch_losses: List[float] = []
    num_samples = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            dataloader,
            desc=f"  Hospital {hospital_id} | Epoch {epoch+1}/{num_epochs}",
            leave=False,
        )

        for batch in pbar:
            # NIHDataset with two_view=True returns (view1, view2)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Unpack — use view1 for MAE (the model handles this internally too)
                imgs = batch[0].to(device)
            else:
                imgs = batch.to(device)

            optimizer.zero_grad()

            # MAE forward pass → reconstruction loss
            loss, _, _ = model(imgs)

            # FedProx proximal term: μ/2 * ||w_local - w_global||²
            if is_fedprox and global_params_flat is not None:
                local_params_flat = _flatten_encoder_params(model, device)
                proximal_term = (mu / 2.0) * torch.sum(
                    (local_params_flat - global_params_flat) ** 2
                )
                loss = loss + proximal_term

            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1
            num_samples += imgs.shape[0]
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        scheduler.step()
        mean_epoch_loss = epoch_loss / max(num_batches, 1)
        epoch_losses.append(mean_epoch_loss)

    # Return only encoder weights — decoder stays local
    encoder_weights = model.get_encoder_weights()

    print(
        f"  [Hospital {hospital_id}] SSL training done | "
        f"Samples: {num_samples} | "
        f"Final loss: {epoch_losses[-1]:.4f}"
    )

    return {
        "encoder_weights": encoder_weights,
        "num_samples": num_samples,
        "epoch_losses": epoch_losses,
    }


# ─── Helper: flatten weight tensors into a 1-D vector ───────────────────────

def _flatten_weights(state_dict: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Flatten a state_dict into a single 1-D parameter vector."""
    return torch.cat([
        v.to(device).float().flatten()
        for v in state_dict.values()
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
    ])


def _flatten_encoder_params(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Flatten current encoder parameters of model into a 1-D vector."""
    encoder_state = model.get_encoder_weights()
    return _flatten_weights(encoder_state, device)
