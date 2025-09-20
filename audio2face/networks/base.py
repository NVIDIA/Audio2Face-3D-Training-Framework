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
import torch.nn as nn


class NetworkBaseRegression(nn.Module):
    def __init__(self) -> None:
        super(NetworkBaseRegression, self).__init__()
        self.supported_modes = [
            "train",
            "onnx",
        ]
        self.set_mode("train")

    def set_mode(self, mode: str) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported network mode: {mode}")
        self.mode = mode


class NetworkBaseDiffusion(nn.Module):
    def __init__(self) -> None:
        super(NetworkBaseDiffusion, self).__init__()
        self.supported_modes = [
            "offline",
            "streaming",
            "streaming_stateless",
            "streaming_stateless_output_delta",
            "streaming_stateless_onnx",
            "streaming_stateless_trt",
            "streaming_stateless_output_delta_onnx",
            "streaming_stateless_output_delta_trt",
        ]
        self.set_mode("offline")

    def set_mode(self, mode: str) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported network mode: {mode}")
        self.mode = mode
        self.forward = {
            "offline": self.forward_offline,
            "streaming": self.forward_streaming,
            "streaming_stateless": self.forward_streaming_stateless,
            "streaming_stateless_output_delta": self.forward_streaming_stateless_output_delta,
            "streaming_stateless_onnx": self.forward_streaming_stateless_onnx,
            "streaming_stateless_trt": self.forward_streaming_stateless_trt,
            "streaming_stateless_output_delta_onnx": self.forward_streaming_stateless_output_delta_onnx,
            "streaming_stateless_output_delta_trt": self.forward_streaming_stateless_output_delta_trt,
        }[self.mode]
