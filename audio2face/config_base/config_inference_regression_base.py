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
# Audio2Face Inference Config (Post-processing params, etc)
CONFIG = {
    "input_strength": 1.3,
    "upper_face_smoothing": 0.001,
    "lower_face_smoothing": 0.0023,
    "upper_face_strength": 1.0,
    "lower_face_strength": 1.7,
    "face_mask_level": 0.6,
    "face_mask_softness": 0.0085,
    "source_shot": None,
    "source_frame": None,
    "skin_strength": 1.1,
    "blink_strength": 1.0,
    "lower_teeth_strength": 1.3,
    "lower_teeth_height_offset": -0.1,
    "lower_teeth_depth_offset": 0.25,
    "lip_open_offset": -0.05,
    "tongue_strength": 1.5,
    "tongue_height_offset": 0.2,
    "tongue_depth_offset": -0.3,
    "eyeballs_strength": 1.0,
    "saccade_strength": 0.9,
    "right_eye_rot_x_offset": 0.0,
    "right_eye_rot_y_offset": -2.0,
    "left_eye_rot_x_offset": 0.0,
    "left_eye_rot_y_offset": 2.0,
    "eyelid_open_offset": 0.06,
    "eye_saccade_seed": 0,
}
