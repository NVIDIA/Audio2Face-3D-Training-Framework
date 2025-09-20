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
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def NormRelu(x: torch.Tensor, leak: float = 0.0) -> torch.Tensor:
    coef = (2.0 / (1.0 + leak**2)) ** 0.5
    return F.leaky_relu(x, leak) * coef


def ConcatEmotion(x: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
    """
    x dim: [batch, seq, ...]
    emotion dim: [batch, seq, emo]
    """
    if not torch.jit.is_tracing():
        assert x.shape[0] == emotion.shape[0], "x and emotion shape[0] mismatch"
        assert x.shape[1] == emotion.shape[1], "x and emotion shape[1] mismatch"
    if len(x.shape) == 3:
        return torch.cat((x, emotion), dim=2)
    elif len(x.shape) == 4:
        return torch.cat((x, emotion[..., None].repeat(1, 1, 1, x.shape[3])), dim=2)
    elif len(x.shape) == 5:
        return torch.cat((x, emotion[..., None, None].repeat(1, 1, 1, x.shape[3], x.shape[4])), dim=2)
    else:
        raise ValueError("Unsupported x shape len", x.shape)


def EyeOpenMask(y_pose: torch.Tensor, eye_dist_verts: dict[str, list[int]], eye_dist_threshold: float) -> torch.Tensor:
    """
    y_pose: [batch, seq, verts, 3]
    """
    dim = [1]  # y-axis
    dist = y_pose[:, :, eye_dist_verts["upper"], :][..., dim] - y_pose[:, :, eye_dist_verts["lower"], :][..., dim]
    eye_open_mask = dist.mean(dim=[1, 2, 3]) > eye_dist_threshold
    return eye_open_mask.detach()


class AutoCorr(nn.Module):
    def __init__(self, **params) -> None:
        super().__init__()

        self.num_autocorr = params.get("num_autocorr", 32)
        self.num_windows = params.get("num_windows", 64)
        self.win_len = params.get("win_len", 256)
        self.win_stride = params.get("win_stride", 128)
        self.preemph = params.get("preemph", 0.0)
        self.remove_dc = params.get("remove_dc", True)
        self.win_func = params.get("win_func", "hanning")

        if self.win_func is not None:
            if self.win_func == "hanning":
                self.win_func_vals = torch.tensor(np.hanning(self.win_len)[None, None, None, :])
            elif self.win_func == "kaiser6":
                self.win_func_vals = torch.tensor(np.kaiser(self.win_len, 6)[None, None, None, :])
            else:
                raise ValueError(f"Invalid win_func: {self.win_func}")
        else:
            self.win_func_vals = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim: [batch, seq, value]
        """
        batch, seq_len = x.shape[:2]

        if self.preemph != 0.0:
            x = torch.cat((x[:, :, :1], x[:, :, 1:] - x[:, :, :-1] * self.preemph), dim=2)

        win_idx, sample_idx = np.ogrid[0 : self.num_windows, 0 : self.win_len]
        win_lut = win_idx * self.win_stride + sample_idx  # (window, sample)
        windows = x[:, :, win_lut]  # (batch, seq, window, sample)

        if self.remove_dc:
            windows = windows - torch.mean(windows, dim=3, keepdim=True)

        if self.win_func_vals is not None:
            windows = windows * self.win_func_vals.to(windows)

        windows = windows.view(-1, self.win_len)
        single_ofs = []
        for ofs in range(self.num_autocorr):
            single_ofs.append(torch.sum(windows[:, : self.win_len - ofs] * windows[:, ofs:], dim=1))
        autocorr = torch.stack(single_ofs).t()
        autocorr = autocorr.view(batch, seq_len, 1, self.num_windows, self.num_autocorr)
        autocorr = autocorr * (1.0 / self.win_len)

        return autocorr


class GDropout(nn.Module):
    def __init__(
        self,
        mode: str = "mul",
        strength: float = 0.4,
        axes: tuple[int, ...] = (0, 2),
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.strength = strength
        self.axes = axes
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        rnd_shape = [x.shape[i] for i in self.axes]
        broadcast = [x.shape[i] if i in self.axes else 1 for i in range(len(x.shape))]

        if self.mode == "drop":
            p = 1.0 - self.strength
            rnd = torch.distributions.Bernoulli(probs=p).sample(rnd_shape) / p
        elif self.mode == "mul":
            rnd = (1.0 + self.strength) ** torch.randn(rnd_shape)
        else:
            raise ValueError("Invalid GDrop mode", self.mode)

        rnd = rnd.to(x)

        if self.normalize:
            rnd = rnd / (1e-6 + torch.mean(rnd**2, dim=1, keepdim=True) ** 0.5)

        return x * rnd.view(broadcast)


class ImplicitEmotionDB(nn.Module):
    def __init__(self, num_elements: int, emo_len: int, init_sigma: float = 1.0) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_elements, emo_len) * init_sigma)

    def forward(self, global_frame_idx: torch.Tensor) -> torch.Tensor:
        return self.W[global_frame_idx, :]


class CoeffsToPose(nn.Module):
    def __init__(self, shapes_matrix: torch.Tensor, shapes_mean: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("shapes_matrix", shapes_matrix)
        self.register_buffer("shapes_mean", shapes_mean)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        pose = torch.matmul(coeffs, self.shapes_matrix.view(self.shapes_matrix.shape[0], -1))
        pose = pose.view(pose.shape[:-1] + (-1, 3)) + self.shapes_mean
        return pose


def linear1d(
    data: torch.Tensor,
    out_length: int,
    align_corners: bool = False,
    half_pixel_centers: bool = True,
) -> torch.Tensor:
    if align_corners and half_pixel_centers:
        raise ValueError("align_corners and half_pixel_centers may not be enabled simultaneously")

    device = data.device

    # Extract data dimensions
    batches, channels, length = data.shape

    # Compute scaling factor - handle length being int or tensor during export
    if isinstance(length, int):
        length_tensor = torch.tensor(length, dtype=torch.float32, device=device)
    else:
        length_tensor = length.float()

    if align_corners:
        scale = torch.clip((length_tensor - 1.0) / (out_length - 1.0), min=0.0)
    else:
        scale = torch.clip(length_tensor / out_length, min=0.0)

    # Get indices for the output
    indices = torch.arange(out_length, device=device)

    # Apply half-pixel offset
    if half_pixel_centers:
        x_in = ((indices + 0.5) * scale) - 0.5
    else:
        x_in = indices * scale

    # Floor the half-pixel offset for weight and coordinate computations
    x_in_floor = torch.floor(x_in)

    # Compute weights needed for interpolation
    x_weight = x_in - x_in_floor

    # Get lower and upper coordinates
    x_lower = torch.clip(x_in_floor, min=0).to(torch.int64)
    if isinstance(length, int):  # Trick to solve issue when during trt export, the length var is somehow a torch tensor
        x_upper = torch.clip(torch.ceil(x_in), max=length - 1).to(torch.int64)
    else:
        x_upper = torch.clip(torch.ceil(x_in), max=length.cuda() - 1).to(torch.int64)

    # Reshaping for efficient computation
    data_flat = data.reshape(batches * channels, length)

    # Calculate contributions from lower and upper points
    a = data_flat[:, x_lower] * (1 - x_weight)
    b = data_flat[:, x_upper] * x_weight

    # Combine contributions and reshape
    return (a + b).reshape(batches, channels, out_length)
