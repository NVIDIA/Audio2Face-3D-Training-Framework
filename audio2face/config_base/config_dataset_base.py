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
########################################################################################################################
# Paths to various parts of the Audio2Face dataset (per actor)
# Check the details for each of the parts in Audio2Face-3D-Dataset-v1.0.0-claire/docs/README.html file
# Actor-specific parameters are represented as a dictionary: PARAM = {"actor1": value1, "actor2": value2, ...}
########################################################################################################################

# Audio data
AUDIO_ROOT = {}

# Skin data
SKIN_CACHE_ROOT = {}
SKIN_NEUTRAL_FPATH = {}
SKIN_NEUTRAL_INFERENCE_FPATH = {}
SKIN_LIP_OPEN_POSE_DELTA_FPATH = {}
SKIN_EYE_CLOSE_POSE_DELTA_FPATH = {}
SKIN_LIP_DIST_VERTEX_LIST_FPATH = {}
SKIN_LIP_SIZE_VERTEX_LIST_FPATH = {}
SKIN_EYE_DIST_VERTEX_LIST_FPATH = {}

# Tongue data
TONGUE_CACHE_ROOT = {}
TONGUE_NEUTRAL_FPATH = {}
TONGUE_NEUTRAL_INFERENCE_FPATH = {}
TONGUE_RIGID_VERTEX_LIST_FPATH = {}

# Jaw data
JAW_KEYPOINTS_NEUTRAL_FPATH = {}
JAW_ANIM_DATA_FPATH = {}

# Eye data
EYE_ANIM_DATA_FPATH = {}
EYE_BLINK_KEYS_FPATH = {}
EYE_SACCADE_ROTATIONS_FPATH = {}

# Blendshape data
BLENDSHAPE_SKIN_FPATH = {}
BLENDSHAPE_SKIN_CONFIG_FPATH = {}
BLENDSHAPE_TONGUE_FPATH = {}
BLENDSHAPE_TONGUE_CONFIG_FPATH = {}

########################################################################################################################
# Dataset properties and meta-information
########################################################################################################################

# List of emotions used in the training dataset shots with facial animation performance
SHOT_EMOTION_NAMES = [
    "neutral",
]

# By default, shot emotion name is inferred from the shot name automatically (if one is a substring of the other)
# This list overrides the emotion names for the specified shots
# Item format: (actor_name, shot_name, shot_emotion_name)
# Example: ("mark", "shot2", "sadness")
SHOT_EMOTION_MAP = []

# Frames-Per-Second rate for all the animation caches in the training dataset (per actor)
# This parameter will be used to generate the shot list artifact during Preprocessing and augmented muted shots
CACHE_FPS = {}

# List of the names of the actors performing the animation in the shots
ACTOR_NAMES = []

# Data transform scale: adjust these values according to your dataset, some data may require increasing scale value
# Format: {"actor_name": {"channel_name": scale_value}}
TRANSFORM_SCALE = {}
