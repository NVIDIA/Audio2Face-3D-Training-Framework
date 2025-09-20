# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
import torch.nn.functional as F


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(y_pred, y_true)
    return loss


def motion_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    mv_true = y_true[:, 1:, ...] - y_true[:, :-1, ...]
    mv_pred = y_pred[:, 1:, ...] - y_pred[:, :-1, ...]
    loss = F.mse_loss(mv_pred, mv_true)
    return loss


def motion_reg_loss(y: torch.Tensor) -> torch.Tensor:
    mv = y[:, 1:, ...] - y[:, :-1, ...]
    loss = torch.mean(mv**2) / torch.mean(y**2)
    return loss


def lip_dist_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    exp_scale: float,
    lip_dist_verts: dict[str, list[int]],
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    y: [batch, seq, verts, 3]
    """
    dim = [1]  # y-axis
    dist_true = y_true[:, :, lip_dist_verts["upper"], :][..., dim] - y_true[:, :, lip_dist_verts["lower"], :][..., dim]
    dist_pred = y_pred[:, :, lip_dist_verts["upper"], :][..., dim] - y_pred[:, :, lip_dist_verts["lower"], :][..., dim]
    dist_weight = torch.exp(-exp_scale * torch.clamp(dist_true, min=0))
    loss = F.mse_loss(dist_weight * dist_pred, dist_weight * dist_true, reduction="none")
    if sample_weights is not None:
        loss = loss * sample_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return loss.mean()


def lip_size_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    lip_size_verts: dict[str, list[int]],
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    y: [batch, seq, verts, 3]
    """
    dim = [1]  # y-axis
    upper_lip_size_true = (
        y_true[:, :, lip_size_verts["upper_lip"]["top"], :][..., dim]
        - y_true[:, :, lip_size_verts["upper_lip"]["bottom"], :][..., dim]
    )
    lower_lip_size_true = (
        y_true[:, :, lip_size_verts["lower_lip"]["top"], :][..., dim]
        - y_true[:, :, lip_size_verts["lower_lip"]["bottom"], :][..., dim]
    )
    upper_lip_size_pred = (
        y_pred[:, :, lip_size_verts["upper_lip"]["top"], :][..., dim]
        - y_pred[:, :, lip_size_verts["upper_lip"]["bottom"], :][..., dim]
    )
    lower_lip_size_pred = (
        y_pred[:, :, lip_size_verts["lower_lip"]["top"], :][..., dim]
        - y_pred[:, :, lip_size_verts["lower_lip"]["bottom"], :][..., dim]
    )
    loss = F.mse_loss(upper_lip_size_pred, upper_lip_size_true, reduction="none")
    loss += F.mse_loss(lower_lip_size_pred, lower_lip_size_true, reduction="none")
    if sample_weights is not None:
        loss = loss * sample_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return loss.mean()


def vol_stab_reg_loss(
    x: torch.Tensor,
    y_pred: torch.Tensor,
    exp_scale: float,
    mask: list[int] | None = None,
    y_skin_pose_pred: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Idea: when volume is low, the PCA coeffs should have smaller change between neighboring frames.
    This encourages less jitter on low volume.
    x: [batch, seq, buf_len] (raw audio or better max normalized to 1)
    y_pred: [batch, seq, target_len]
    """
    mean_volume = x.abs().mean(dim=[-1, -2])  # [batch]
    volume_weight = torch.exp(-exp_scale * mean_volume)  # larger weight for small volume
    if mask is not None and y_skin_pose_pred is not None:  # calculate loss on masked skin region
        y_skin_pose_pred = y_skin_pose_pred[:, :, mask, :]
        mv = y_skin_pose_pred[:, 1:, ...] - y_skin_pose_pred[:, :-1, ...]  # [batch, seq-1, verts, 3]
        loss = (mv**2).mean(dim=[1, 2, 3])
    else:
        mv = y_pred[:, 1:, :-4] - y_pred[:, :-1, :-4]  # [batch, seq-1, targets], ignore eye values (:-4)
        loss = (mv**2).mean(dim=[1, 2])  # [batch]
    loss = (volume_weight * loss).mean()
    return loss


def phoneme_loss(
    p_pred_all: list[torch.Tensor],
    p_true_all: list[torch.Tensor],
    p_weights_all: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    p_pred_all: list of predicted phoneme tensors for different languages
    p_true_all: list of ground-truth phoneme tensors for different languages
    p_weights_all: list of phoneme importance weights for different languages
    """
    if p_weights_all is None:
        p_weights_all = [None] * len(p_pred_all)
    loss = 0
    for p_pred, p_true, p_weights in zip(p_pred_all, p_true_all, p_weights_all):
        loss += F.cross_entropy(
            p_pred.view(-1, p_pred.shape[-1]),
            p_true.view(-1, p_true.shape[-1]),
            weight=p_weights,
        )
    return loss


def phoneme_motion_loss(
    p_pred_all: list[torch.Tensor],
    p_true_all: list[torch.Tensor],
) -> torch.Tensor:
    """
    p_pred_all: list of predicted phoneme tensors for different languages
    p_true_all: list of ground-truth phoneme tensors for different languages
    """
    loss = 0
    for p_pred, p_true in zip(p_pred_all, p_true_all):
        loss += motion_loss(p_pred, p_true)
    return loss


def cal_mask_params(
    mean_face: torch.Tensor,
    eye_dist_verts: dict[str, list[int]],
) -> tuple[float, float, float]:
    """
    Estimate the parameters for the upper face mask
    """
    lower_eye_vert_1 = mean_face[eye_dist_verts["lower"][0]]
    lower_eye_vert_2 = mean_face[eye_dist_verts["lower"][1]]
    y_thres = min(lower_eye_vert_1[1], lower_eye_vert_2[1]).item()  # y_dim (height)

    LR_eye_dist = torch.norm(lower_eye_vert_1 - lower_eye_vert_2)
    eye_z = (lower_eye_vert_1[2] + lower_eye_vert_2[2]) / 2
    z_thres = (eye_z - LR_eye_dist).item()
    sharpness = (LR_eye_dist / 2).item()
    return y_thres, z_thres, sharpness


def generate_vertex_weights_vectorized(
    vertices: torch.Tensor,
    y_thres: float = 174.5,
    z_thres: float | None = 7.0,
    sharpness: float = 3.0,
) -> torch.Tensor:
    y_coords = vertices[:, 1]  # shape = (num_vertices,)
    z_coords = vertices[:, 2]
    if z_thres is None:  # Use upper face mask
        weights = 1.0 / (1.0 + torch.exp(sharpness * (y_thres - y_coords)))  # Vx1
    else:  # use upper/front face mask
        weights = (
            1.0
            / (1.0 + torch.exp(sharpness * (y_thres - y_coords)))
            * 1.0
            / (1.0 + torch.exp(sharpness * (z_thres - z_coords)))
        )  # Vx1
    return weights


def expression_smooth_reg(y: torch.Tensor, softmask_upperface: torch.Tensor) -> torch.Tensor:
    """
    input is N,seq,Vx3
    """
    N, seq = y.shape[:2]
    y = y.reshape(N, seq, -1, 3)
    mv = y[:, 1:, :, :] - y[:, :-1, :, :]
    loss = torch.mean(softmask_upperface * (mv**2))
    return loss


def r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    y_true_mean = torch.mean(y_true, axis=(0, 1))
    num = torch.mean((y_pred - y_true) ** 2)
    denom = torch.mean((y_true_mean - y_true) ** 2)
    return 1.0 - num / denom


class LossNormalizer:
    def __init__(self, mode: str, decay: float) -> None:
        self.mode = mode
        self.decay = decay
        self.num = 0.0
        self.den = 0.0

    def reset(self) -> None:
        self.num = 0.0
        self.den = 0.0

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            self.num = self.num * self.decay + loss.item()
            self.den = self.den * self.decay + 1.0
            ratio = self.num / self.den
            return loss / (ratio + 1.0e-8)

        elif self.mode == "stddev":
            self.num = self.num * self.decay + loss.item() ** 2
            self.den = self.den * self.decay + 1.0
            ratio = self.num / self.den
            return loss / (ratio**0.5 + 1.0e-8)
