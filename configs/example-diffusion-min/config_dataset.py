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
# Only AUDIO_ROOT and SKIN_CACHE_ROOT are required - all other paths are optional
########################################################################################################################

# Audio data
AUDIO_ROOT = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/audio",
}

# Skin data
SKIN_CACHE_ROOT = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/cache/skin",
}

########################################################################################################################
# Dataset properties and meta-information
########################################################################################################################

# List of emotions used in the training dataset shots with facial animation performance
SHOT_EMOTION_NAMES = [
    "neutral",
    "amazement",
    "anger",
    "cheekiness",
    "disgust",
    "fear",
    "grief",
    "joy",
    "outofbreath",
    "pain",
    "sadness",
]

# Frames-Per-Second rate for all the animation caches in the training dataset (per actor)
# This parameter will be used to generate the shot list artifact during Preprocessing and augmented muted shots
CACHE_FPS = {
    "claire": 30.0,
}

# List of the names of the actors performing the animation in the shots
ACTOR_NAMES = [
    "claire",
]

# Data transform scale: adjust these values according to your dataset, some data may require increasing scale value
# Format: {"actor_name": {"channel_name": scale_value}}
TRANSFORM_SCALE = {
    "claire": {
        "skin": 1.0,
        "tongue": 1.0,
        "jaw": 1.0,
        "eye": 1.0,
    },
}
