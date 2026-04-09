"""
src/models/proto_head.py
------------------------
Prototypical Network classification head for few-shot TB detection.

Prototypical Networks:
  - Compute a prototype (mean embedding) per class from the support set
  - Classify query samples by nearest prototype (Euclidean distance)
  - No learnable parameters beyond optional fine-tuning MLP

Reference: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PrototypicalHead(nn.Module):
    """
    Prototypical Network classification head for binary TB/Normal classification.

    Supports two inference modes:
        1. Prototypical (default): classify by minimum Euclidean distance to prototypes
        2. Linear (fallback)     : standard cross-entropy fine-tuning

    Args:
        embed_dim   : Encoder output dimensionality (e.g., 512)
        num_classes : Number of classes (default 2: Normal=0, TB=1)
        use_linear  : If True, adds a trainable linear head for CE fine-tuning
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 2,
        use_linear: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_linear = use_linear

        # Optional linear head for cross-entropy fine-tuning fallback
        if use_linear:
            self.linear_head = nn.Linear(embed_dim, num_classes)

        # Trainable projection layer (to allow fine-tuning the embedding space)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Stored prototypes (set during few-shot episode)
        self.register_buffer("prototypes", torch.zeros(num_classes, embed_dim))
        self._prototypes_computed = False

    # ─── Prototype Computation ───────────────────────────────────────────────

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute class prototypes as mean embedding of support samples per class.

        Args:
            support_embeddings : (N_support, embed_dim)
            support_labels     : (N_support,) — integer class labels {0, 1, ..., C-1}

        Returns:
            prototypes : (num_classes, embed_dim)
        """
        prototypes = torch.zeros(
            self.num_classes, self.embed_dim,
            device=support_embeddings.device,
            dtype=support_embeddings.dtype,
        )
        for c in range(self.num_classes):
            mask = (support_labels == c)
            if mask.sum() == 0:
                # No samples for this class — leave prototype as zero
                continue
            prototypes[c] = support_embeddings[mask].mean(dim=0)

        # Store for use in forward()
        self.prototypes = prototypes.detach() # don't backprop into prototypes directly
        self._prototypes_computed = True
        return prototypes

    def get_learnable_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Identical to compute_prototypes but allows grad through the mean."""
        # Project support embeddings before computing mean
        support_embeddings = self.projection(support_embeddings)
        
        prototypes = torch.zeros(
            self.num_classes, self.embed_dim,
            device=support_embeddings.device,
            dtype=support_embeddings.dtype,
        )
        for c in range(self.num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                prototypes[c] = support_embeddings[mask].mean(dim=0)
        return prototypes

    # ─── Prototypical Forward ────────────────────────────────────────────────

    def forward(
        self,
        query_embeddings: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Classify query samples using prototypical distance.

        Args:
            query_embeddings : (B, embed_dim) query embeddings
            prototypes       : (num_classes, embed_dim) — if None, use stored prototypes

        Returns:
            logits : (B, num_classes) — softmax-normalized scores
                     (negative squared Euclidean distance → softmax)
        """
        if prototypes is None:
            if not self._prototypes_computed:
                raise RuntimeError(
                    "Prototypes not computed. Call compute_prototypes() first "
                    "or pass prototypes explicitly."
                )
            prototypes = self.prototypes

        # Project query embeddings into the metric space
        query_embeddings = self.projection(query_embeddings)

        return self._prototypical_logits(query_embeddings, prototypes)

    def _prototypical_logits(
        self,
        queries: torch.Tensor,
        protos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative squared Euclidean distances, convert to probabilities via softmax.

        Args:
            queries : (B, D)
            protos  : (C, D)
        Returns:
            probs : (B, C) — softmax probabilities
        """
        # Broadcast: (B, 1, D) - (1, C, D) → (B, C, D)
        diffs = queries.unsqueeze(1) - protos.unsqueeze(0)  # (B, C, D)
        sq_dists = (diffs ** 2).sum(dim=-1)                  # (B, C)
        logits = -sq_dists                                   # negative distance as logit
        probs = F.softmax(logits, dim=-1)                    # (B, C)
        return probs

    # ─── Linear CE Forward ───────────────────────────────────────────────────

    def linear_forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Standard linear classification (cross-entropy fallback).

        Args:
            embeddings : (B, embed_dim)
        Returns:
            logits : (B, num_classes)
        """
        if not self.use_linear:
            raise RuntimeError("Linear head not instantiated. Set use_linear=True.")
        return self.linear_head(embeddings)

    # ─── Loss Helpers ─────────────────────────────────────────────────────────

    def prototypical_loss(
        self,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-entropy loss over prototypical probabilities.

        Returns:
            loss    : scalar CE loss
            probs   : (B, num_classes) probabilities
        """
        probs = self.forward(query_embeddings, prototypes)
        # Cross-entropy on log-probabilities
        log_probs = torch.log(probs + 1e-8)
        loss = F.nll_loss(log_probs, query_labels)
        return loss, probs

    def linear_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard cross-entropy loss with linear head.

        Returns:
            loss   : scalar CE loss
            logits : (B, num_classes)
        """
        logits = self.linear_forward(embeddings)
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    # ─── Utility ──────────────────────────────────────────────────────────────

    def predict(
        self,
        query_embeddings: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels and probabilities.

        Returns:
            predicted_labels : (B,) int tensor
            probs            : (B, num_classes) probability tensor
        """
        probs = self.forward(query_embeddings, prototypes)
        labels = probs.argmax(dim=-1)
        return labels, probs
