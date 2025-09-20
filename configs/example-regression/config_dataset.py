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
AUDIO_ROOT = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/audio",
}

# Skin data
SKIN_CACHE_ROOT = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/cache/skin",
}
SKIN_NEUTRAL_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/geom/skin/neutral_pose.npy",
}
SKIN_LIP_OPEN_POSE_DELTA_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/geom/skin/lip_open_pose_delta.npy",
}
SKIN_EYE_CLOSE_POSE_DELTA_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/geom/skin/eye_close_pose_delta.npy",
}
SKIN_LIP_DIST_VERTEX_LIST_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/geom/skin/lip_dist_vertex_list.json",
}
SKIN_LIP_SIZE_VERTEX_LIST_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/geom/skin/lip_size_vertex_list.json",
}
SKIN_EYE_DIST_VERTEX_LIST_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/geom/skin/eye_dist_vertex_list.json",
}

# Tongue data
TONGUE_CACHE_ROOT = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/cache/tongue",
}
TONGUE_NEUTRAL_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/geom/tongue/neutral_pose.npy",
}
TONGUE_RIGID_VERTEX_LIST_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/geom/tongue/rigid_vertex_list.json",
}

# Jaw data
JAW_KEYPOINTS_NEUTRAL_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/xform/jaw_keypoints_neutral.npy",
}
JAW_ANIM_DATA_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/xform/jaw_keypoints_cache_all.npz",
}

# Eye data
EYE_ANIM_DATA_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/xform/eye_rotations_all.npz",
}
EYE_BLINK_KEYS_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/xform/eye_blink_keys.npy",
}
EYE_SACCADE_ROTATIONS_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/xform/eye_saccade_rotations.npy",
}

# Blendshape data
BLENDSHAPE_SKIN_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/bs_data/bs_skin.npz",
}
BLENDSHAPE_SKIN_CONFIG_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/bs_data/bs_skin_config.json",
}
BLENDSHAPE_TONGUE_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/bs_data/bs_tongue.npz",
}
BLENDSHAPE_TONGUE_CONFIG_FPATH = {
    "claire": "/datasets/Audio2Face-3D-Dataset-v1.0.0-claire/data/claire/bs_data/bs_tongue_config.json",
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
