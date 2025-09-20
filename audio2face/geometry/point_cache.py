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
import struct
import numpy as np


def read_cache_pc2(fpath: str) -> np.ndarray:
    header_format = "<12siiffi"
    with open(fpath, "rb") as f:
        header_size = struct.calcsize(header_format)
        header = f.read(header_size)
        signature, file_version, num_verts, start_frame, sample_rate, num_frames = struct.unpack(header_format, header)
        if signature.decode() != "POINTCACHE2\0":
            raise ValueError(f"Invalid pc2 file: {fpath}")
        data = np.fromfile(f, dtype=np.float32, count=num_frames * num_verts * 3)
        data = data.reshape(num_frames, num_verts, 3)
    return data


def write_cache_pc2(fpath: str, data: np.ndarray, sample_rate: float = 1.0) -> None:
    header_format = "<12siiffi"
    if len(data.shape) != 3:
        raise ValueError(f"Invalid data shape: {data.shape}, must be (num_frames, num_vets, 3)")
    num_frames = data.shape[0]
    num_verts = data.shape[1]
    start_frame = 0.0
    file_version = 1
    signature = b"POINTCACHE2\0"
    header = struct.pack(header_format, signature, file_version, num_verts, start_frame, sample_rate, num_frames)
    with open(fpath, "wb") as f:
        f.write(header)
        data.astype("<f").tofile(f)
