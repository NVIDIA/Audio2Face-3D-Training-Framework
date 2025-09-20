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
# The name of the Preprocessing run (use different names for different hyper-parameters or datasets)
RUN_NAME = "default"

# Additional information describing the Preprocessing run, will be saved to <PREPROC_RUN_NAME_FULL>/configs/info.txt
RUN_INFO = ""

########################################################################################################################
# Skin preproc params
########################################################################################################################

# Path to the skin cache directory with different resolution, organized the same way as the data in SKIN_CACHE_ROOT
# Lower resolution cache can be used in Preprocessing pruning for reducing memory usage and computation time
# Shot list and shot lengths should match the data in SKIN_CACHE_ROOT
# If the value for an actor is missing -> Preprocessing pruning will use the original data from SKIN_CACHE_ROOT
SKIN_PRUNE_CACHE_ROOT = {}

# Path to the prune_mesh_mask.npy file with a subset of skin mesh vertices (mask), covering only moving parts
# This mask can be used in Preprocessing pruning for reducing memory usage and computation time
# The mask is a 1D numpy array containing indices of the skin mesh vertices corresponding to the mask
# Dimensions: [num_mask_vertices], data type: int
# If SKIN_PRUNE_CACHE_ROOT is used, the same mesh resolution / topology should be used for SKIN_PRUNE_MESH_MASK
# If the value for an actor is missing -> Preprocessing pruning will use all skin mesh vertices (no mask)
SKIN_PRUNE_MESH_MASK_FPATH = {}

# If the value for an actor is missing -> Use all shots in the directory
SKIN_CACHE_SHOTS = {}

# If the value for an actor is missing -> Use all shots in the directory
SKIN_PRUNE_CACHE_SHOTS = {}

# If the value for an actor is missing -> Use variance threshold to automatically infer the number of components
SKIN_FORCE_COMPONENTS = {}

SKIN_PRUNE_SIM_DIST = 4.0
SKIN_SELECT_DISTINCT_MAX_ITER = 787
SKIN_PCA_VARIANCE_THRESHOLD = 0.9995

########################################################################################################################
# Tongue preproc params
########################################################################################################################

# If the value for an actor is missing -> Use all shots in the directory
TONGUE_CACHE_SHOTS = {}

# If the value for an actor is missing -> Use variance threshold to automatically infer the number of components
TONGUE_FORCE_COMPONENTS = {}

TONGUE_PCA_VARIANCE_THRESHOLD = 0.9995

########################################################################################################################
# Default preproc artifact dimensions
########################################################################################################################

# Default jaw keypoints shape for neutral jaw if no data is provided
DEFAULT_JAW_KEYPOINTS_SHAPE = (5, 3)

# Default eye blink keys if no data is provided
DEFAULT_EYE_BLINK_KEYS_SHAPE = (10,)

# Default eye saccade rotations if no data is provided
DEFAULT_EYE_SACCADE_ROT_SHAPE = (5000, 2)

# Default tongue PCA shape if no data is provided
DEFAULT_TONGUE_PCA_SHAPE = (10, 520, 3)

########################################################################################################################
# Misc
########################################################################################################################

# This location is used to write output Preprocessing artifacts
PREPROC_OUTPUT_ROOT = "/workspace/output_preproc"

VERBOSE = False
