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
import cupy as cp
from cupy_backends.cuda.libs.cusolver import CUSOLVERError


def pca_truncated(
    data: np.ndarray,
    variance_threshold: float,
    custom_mean: np.ndarray | None = None,
    force_components: int | None = None,
    use_cupy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates PCA decomposition and truncates the result to have specific number of components.
    Number of components is determined by variance threshold and additionally (optionally) by the provided number.

    Parameters
    ----------
    data : NumPy or cupy array
        Input data
    variance_threshold : float
        Variance threshold for determining number of components to truncate the result
    custom_mean : NumPy or cupy array, optional
        Tensor with user-defined custom mean to be subtracted from the data
    force_components : int
        Additional user-defined forced number of components to truncate the result
    use_cupy : bool
        If true -- use cupy backend for SVD and linear algebra, else -- use NumPy

    Returns
    -------
    NumPy or cupy array: Eigen vectors (truncated)
    NumPy or cupy array: Eigen values (truncated)
    NumPy or cupy array: Mean of the data (calculated or user-defined)

    """

    m, _ = data.shape
    if custom_mean is None:
        if use_cupy:
            mean = cp.mean(data, axis=0)
        else:
            mean = np.mean(data, axis=0)
    else:
        mean = custom_mean
    deltaData = data - mean

    if use_cupy:
        try:
            U, s, VT = cp.linalg.svd(deltaData, full_matrices=0)
        except CUSOLVERError:
            raise RuntimeError(
                f"Unable to compute SVD for a matrix with shape {deltaData.shape}, probably due to running out of GPU memory"
            ) from None
    else:
        U, s, VT = np.linalg.svd(deltaData, full_matrices=0)

    s = s * s / (m - 1)
    evals = s
    evecs = VT.T

    if use_cupy:
        evalRatio = cp.asnumpy(cp.cumsum(evals) / cp.sum(evals))
    else:
        evalRatio = np.cumsum(evals) / np.sum(evals)
    t = np.argwhere(evalRatio > variance_threshold)[0]
    num_components = t[0] + 1

    if force_components is None:
        evecs_t = evecs[:, :num_components]
        evals_t = evals[:num_components]
    else:
        evecs_t = evecs[:, :force_components]
        evals_t = evals[:force_components]

    return evecs_t, evals_t, mean.flatten()
