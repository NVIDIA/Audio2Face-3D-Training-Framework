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


def rigidXform(aPose: np.ndarray, bPose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aMean = np.mean(aPose, axis=0)
    aDelta = aPose - aMean

    bMean = np.mean(bPose, axis=0)
    bDelta = bPose - bMean

    H = np.dot(bDelta.T, aDelta)
    U, s, V = np.linalg.svd(H)

    R = np.dot(V.T, U.T)
    eye = np.eye(3)
    eye[2, 2] = np.linalg.det(R)
    R = np.dot(np.dot(V.T, eye), U.T)

    RR = R.T
    tt = aMean - np.dot(bMean, R.T)

    return RR, tt
