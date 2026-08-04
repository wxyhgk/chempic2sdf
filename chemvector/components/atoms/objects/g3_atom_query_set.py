"""Permutation-invariant Atom queries over a frozen high-recall peak set."""

from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class G3AtomQuerySetConfig:
    candidates: int = 128
    queries: int = 128
    feature_dim: int = 32
    d_model: int = 64
    heads: int = 4
    ffn_dim: int = 192
    encoder_layers: int = 2
    decoder_layers: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.candidates,
            self.queries,
            self.feature_dim,
            self.d_model,
            self.heads,
            self.ffn_dim,
            self.encoder_layers,
            self.decoder_layers,
        ) < 1:
            raise ValueError("G3 Atom query-set dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("G3 Atom query-set heads must divide d_model")
        if self.queries > self.candidates:
            raise ValueError("G3 Atom queries cannot exceed candidate pointers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("G3 Atom query-set dropout is invalid")


def _validate_candidates(candidates: dict[str, torch.Tensor], config) -> int:
    required = {
        "candidate_features",
        "candidate_anchors",
        "raw_scores",
        "normalized_scores",
        "scale_ranks",
    }
    missing = required - candidates.keys()
    if missing:
        raise ValueError(f"G3 Atom query candidates are missing: {sorted(missing)}")
    anchors = candidates["candidate_anchors"]
    if anchors.ndim != 3 or anchors.shape[1:] != (config.candidates, 2):
        raise ValueError("G3 Atom candidate anchors must have shape [batch,128,2]")
    batch = anchors.size(0)
    expected = {
        "candidate_features": (batch, config.candidates, config.feature_dim),
        "raw_scores": (batch, config.candidates),
        "normalized_scores": (batch, config.candidates),
        "scale_ranks": (batch, config.candidates),
    }
    for name, shape in expected.items():
        if candidates[name].shape != shape:
            raise ValueError(f"G3 Atom query candidate {name} has invalid shape")
    devices = {value.device for value in candidates.values()}
    if len(devices) != 1:
        raise ValueError("G3 Atom query candidates must share one device")
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in candidates.values()
    ):
        raise ValueError("G3 Atom query candidates must be finite")
    return batch


class G3AtomQuerySet(nn.Module):
    """Predict Atom existence, count, and a unique candidate pointer as one set."""

    def __init__(self, config: G3AtomQuerySetConfig | None = None) -> None:
        super().__init__()
        self.config = config or G3AtomQuerySetConfig()
        config = self.config
        self.feature_norm = nn.LayerNorm(config.feature_dim, elementwise_affine=False)
        self.feature_projection = nn.Linear(config.feature_dim, config.d_model)
        self.geometry_projection = nn.Sequential(
            nn.Linear(5, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.query_embedding = nn.Embedding(config.queries, config.d_model)
        self.query_reference = nn.Embedding(config.queries, 2)
        self.reference_projection = nn.Linear(2, config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.pointer_query = nn.Linear(config.d_model, config.d_model)
        self.pointer_key = nn.Linear(config.d_model, config.d_model)
        self.existence_head = nn.Linear(config.d_model, 1)
        self.count_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.queries + 1),
        )
        self._initialize_references()

    def _initialize_references(self) -> None:
        side = math.ceil(math.sqrt(self.config.queries))
        positions = []
        for index in range(self.config.queries):
            row, column = divmod(index, side)
            positions.append(((column + 0.5) / side, (row + 0.5) / side))
        with torch.no_grad():
            self.query_reference.weight.copy_(torch.tensor(positions))

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def forward(self, candidates: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = _validate_candidates(candidates, self.config)
        features = candidates["candidate_features"].detach().float()
        anchors = candidates["candidate_anchors"].detach().float()
        geometry = torch.cat(
            (
                anchors,
                candidates["raw_scores"].detach().float().unsqueeze(-1),
                candidates["normalized_scores"].detach().float().unsqueeze(-1),
                candidates["scale_ranks"].detach().float().unsqueeze(-1),
            ),
            dim=-1,
        )
        memory = self.encoder(
            self.feature_projection(self.feature_norm(features))
            + self.geometry_projection(geometry)
        )
        query = (
            self.query_embedding.weight
            + self.reference_projection(self.query_reference.weight)
        ).unsqueeze(0).expand(batch, -1, -1)
        hidden = self.decoder(query, memory)
        pointer_logits = torch.matmul(
            self.pointer_query(hidden), self.pointer_key(memory).transpose(1, 2)
        ) / math.sqrt(self.config.d_model)
        pointer_probabilities = pointer_logits.float().softmax(dim=-1)
        coordinates = torch.matmul(pointer_probabilities, anchors)
        return {
            "existence_logits": self.existence_head(hidden).squeeze(-1).float(),
            "count_logits": self.count_head(memory.mean(dim=1)).float(),
            "pointer_logits": pointer_logits.float(),
            "coordinates": coordinates.float(),
        }


@torch.no_grad()
def hungarian_query_targets(
    output: dict[str, torch.Tensor],
    target_anchors: torch.Tensor,
    target_mask: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    coordinates = output["coordinates"].detach().float().cpu()
    existence = output["existence_logits"].detach().float().sigmoid().cpu()
    targets = target_anchors.detach().float().cpu()
    masks = target_mask.detach().cpu()
    assignments = []
    for batch_index in range(len(coordinates)):
        target_indices = torch.where(masks[batch_index])[0]
        expected = targets[batch_index, target_indices]
        if not len(expected):
            raise ValueError("G3 Atom query matching requires at least one target")
        cost = 5.0 * torch.cdist(coordinates[batch_index], expected, p=1)
        cost = cost - existence[batch_index, :, None]
        queries, columns = linear_sum_assignment(cost.numpy())
        matched_targets = target_indices[torch.as_tensor(columns, dtype=torch.long)]
        assignments.append((
            torch.as_tensor(queries, dtype=torch.long, device=target_anchors.device),
            matched_targets.to(target_anchors.device),
        ))
    return assignments


def g3_atom_query_set_loss(
    output: dict[str, torch.Tensor],
    candidate_anchors: torch.Tensor,
    target_anchors: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    radius_pixels: float = 6.0,
    image_size: int = 768,
) -> dict[str, torch.Tensor]:
    existence = output["existence_logits"]
    count_logits = output["count_logits"]
    pointer_logits = output["pointer_logits"]
    coordinates = output["coordinates"]
    batch, queries = existence.shape
    if count_logits.shape != (batch, queries + 1):
        raise ValueError("G3 Atom query count logits have invalid shape")
    if pointer_logits.shape != (batch, queries, candidate_anchors.size(1)):
        raise ValueError("G3 Atom query pointer logits have invalid shape")
    if coordinates.shape != (batch, queries, 2):
        raise ValueError("G3 Atom query coordinates have invalid shape")
    assignments = hungarian_query_targets(output, target_anchors, target_mask)
    object_targets = torch.zeros_like(existence)
    coordinate_losses = []
    pointer_losses = []
    pointer_log_probabilities = pointer_logits.log_softmax(dim=-1)
    radius = float(radius_pixels) / float(image_size)
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        object_targets[batch_index, query_indices] = 1.0
        expected = target_anchors[batch_index, target_indices]
        coordinate_losses.append(
            F.l1_loss(coordinates[batch_index, query_indices], expected)
        )
        distances = torch.cdist(
            expected.float(), candidate_anchors[batch_index].float()
        )
        compatible = distances <= radius
        if bool((compatible.sum(dim=1) < 1).any()):
            raise RuntimeError("G3 Atom query target has no compatible candidate")
        for offset, query_index in enumerate(query_indices):
            pointer_losses.append(-torch.logsumexp(
                pointer_log_probabilities[batch_index, query_index][compatible[offset]],
                dim=0,
            ))
    object_weights = torch.where(
        object_targets.bool(),
        torch.ones_like(object_targets),
        torch.full_like(object_targets, 0.1),
    )
    object_loss = F.binary_cross_entropy_with_logits(
        existence, object_targets, weight=object_weights, reduction="sum"
    ) / object_weights.sum().clamp_min(1.0)
    coordinate_loss = torch.stack(coordinate_losses).mean()
    pointer_loss = torch.stack(pointer_losses).mean()
    counts = target_mask.sum(dim=1).long()
    count_loss = F.cross_entropy(count_logits, counts)
    loss = object_loss + 5.0 * coordinate_loss + pointer_loss + count_loss
    return {
        "loss": loss,
        "object_loss": object_loss,
        "coordinate_loss": coordinate_loss,
        "pointer_loss": pointer_loss,
        "count_loss": count_loss,
        "matched_queries": loss.new_tensor(float(sum(len(item[0]) for item in assignments))),
    }


@torch.no_grad()
def decode_g3_atom_query_set(
    output: dict[str, torch.Tensor],
    candidate_anchors: torch.Tensor,
    *,
    counts: torch.Tensor | None = None,
) -> list[dict[str, torch.Tensor]]:
    existence = output["existence_logits"].float()
    pointer = output["pointer_logits"].float()
    if counts is None:
        counts = output["count_logits"].argmax(dim=-1)
    if counts.shape != (len(existence),):
        raise ValueError("G3 Atom query decode counts must have shape [batch]")
    decoded = []
    for batch_index in range(len(existence)):
        count = int(counts[batch_index])
        if not 0 <= count <= pointer.size(2):
            raise ValueError("G3 Atom query decoded count is outside candidate capacity")
        if count == 0:
            decoded.append({
                "anchors": candidate_anchors.new_zeros((0, 2)),
                "query_indices": torch.empty(0, dtype=torch.long, device=pointer.device),
                "candidate_indices": torch.empty(0, dtype=torch.long, device=pointer.device),
            })
            continue
        query_indices = existence[batch_index].topk(count).indices
        cost = -pointer[batch_index, query_indices].detach().cpu().numpy()
        rows, columns = linear_sum_assignment(cost)
        row_indices = torch.as_tensor(rows, dtype=torch.long, device=pointer.device)
        candidate_indices = torch.as_tensor(
            columns, dtype=torch.long, device=pointer.device
        )
        selected_queries = query_indices[row_indices]
        decoded.append({
            "anchors": candidate_anchors[batch_index, candidate_indices],
            "query_indices": selected_queries,
            "candidate_indices": candidate_indices,
        })
    return decoded


__all__ = [
    "G3AtomQuerySet",
    "G3AtomQuerySetConfig",
    "decode_g3_atom_query_set",
    "g3_atom_query_set_loss",
    "hungarian_query_targets",
]
