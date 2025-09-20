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
import os
import numpy as np

from audio2face.geometry import maya_cache, point_cache


def read_cache(fpath: str) -> np.ndarray:
    _, ext = os.path.splitext(fpath)
    if ext == ".npy":
        return np.load(fpath)
    elif ext == ".xml":
        return maya_cache.read_cache_mc(fpath)
    elif ext == ".pc2":
        return point_cache.read_cache_pc2(fpath)
    else:
        raise ValueError(f"Unable to read Animation Cache, unrecognized file ext: {ext}")
