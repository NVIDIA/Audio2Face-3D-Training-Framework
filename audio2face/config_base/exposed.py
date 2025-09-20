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
# Config params accessible in partial exposure mode
# For paths to nested params like {A: {B: ...}} use "A/B" notation
EXPOSED_CONFIG_PARAMS = {
    "dataset": [
        "AUDIO_ROOT",
        "SKIN_CACHE_ROOT",
        "SKIN_NEUTRAL_FPATH",
        "SKIN_LIP_OPEN_POSE_DELTA_FPATH",
        "SKIN_EYE_CLOSE_POSE_DELTA_FPATH",
        "SKIN_LIP_DIST_VERTEX_LIST_FPATH",
        "SKIN_EYE_DIST_VERTEX_LIST_FPATH",
        "TONGUE_CACHE_ROOT",
        "TONGUE_NEUTRAL_FPATH",
        "TONGUE_RIGID_VERTEX_LIST_FPATH",
        "JAW_KEYPOINTS_NEUTRAL_FPATH",
        "JAW_ANIM_DATA_FPATH",
        "EYE_ANIM_DATA_FPATH",
        "EYE_BLINK_KEYS_FPATH",
        "EYE_SACCADE_ROTATIONS_FPATH",
        "SHOT_EMOTION_NAMES",
        "SHOT_EMOTION_MAP",
        "CACHE_FPS",
        "ACTOR_NAMES",
    ],
    "preproc": [
        "RUN_NAME",
        "RUN_INFO",
        "SKIN_PRUNE_CACHE_ROOT",
        "SKIN_PRUNE_MESH_MASK_FPATH",
    ],
    "train": [
        "RUN_NAME",
        "RUN_INFO",
        "NETWORK_VERSION",
        "NETWORK_TYPE",
        "PREPROC_RUN_NAME_FULL",
        "CLIPS",
        "AUG_MUTED_SHOTS",
        "NUM_EPOCHS",
        "LOSS_LIP_DIST_ALPHA",
    ],
    "inference": [
        "CONFIG",
    ],
}
