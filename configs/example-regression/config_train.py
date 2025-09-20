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
# The name of the Training run (use different names for different hyper-parameters or datasets)
RUN_NAME = "example-regression"

# Additional information describing the Training run, will be saved to <TRAINING_RUN_NAME_FULL>/configs/info.txt
RUN_INFO = ""

# The version of the final trained network artifact at deployment, stored in network_info.json
NETWORK_VERSION = "2.3"

NETWORK_TYPE = "regression"

########################################################################################################################
# Preprocessing artifacts
########################################################################################################################

# This location is used to read input Preprocessing artifacts for Train and Deploy
PREPROC_ROOT = "/workspace/output_preproc"

# This parameter should contain the full name of the directory with Preprocessing artifacts (per actor)
# Set this parameter after running "run_preproc" and before running "run_train"
# Details about the preproc artifacts are located at <PREPROC_RUN_NAME_FULL>/configs/info.txt
# Actor-specific parameters are represented as a dictionary: PARAM = {"actor1": value1, "actor2": value2, ...}
PREPROC_RUN_NAME_FULL = {
    "claire": "XXXXXX_XXXXXX_example",
}

########################################################################################################################
# Dataset clips
########################################################################################################################

# This list represents the clips which will be used for the Training
# Set this parameter after running "run_preproc" and before running "run_train"
# Several clips could be associated with the same shot (e.g. with different range, augmented audio tracks, etc)
# Use <PREPROC_RUN_NAME_FULL>/deploy/clips_template.py as a template (per actor), modify if needed
# Item format: (actor_name, shot_name, (clip_first_frame, clip_last_frame), speaker_name, audio_lang, audio_fpath_rel, audio_offset)
# Example: ("mark", "shot1", (0, 100), "mark", "en", "shot1.wav", 0)
CLIPS = [
    ("claire", "cp1_neutral", (0, 348), "claire", "zh", "cp1_neutral.wav", 0),  # Shot: [0, 348]
    ("claire", "cp2_neutral", (0, 343), "claire", "zh", "cp2_neutral.wav", 0),  # Shot: [0, 343]
    ("claire", "cp3_neutral", (0, 337), "claire", "zh", "cp3_neutral.wav", 0),  # Shot: [0, 337]
    ("claire", "cp4_neutral", (0, 355), "claire", "zh", "cp4_neutral.wav", 0),  # Shot: [0, 355]
    ("claire", "cp5_neutral", (0, 319), "claire", "zh", "cp5_neutral.wav", 0),  # Shot: [0, 319]
    ("claire", "cp6_neutral", (0, 283), "claire", "zh", "cp6_neutral.wav", 0),  # Shot: [0, 283]
    ("claire", "cp7_neutral", (0, 250), "claire", "zh", "cp7_neutral.wav", 0),  # Shot: [0, 250]
    ("claire", "cp8_neutral", (0, 349), "claire", "zh", "cp8_neutral.wav", 0),  # Shot: [0, 349]
    ("claire", "cp9_neutral", (0, 396), "claire", "zh", "cp9_neutral.wav", 0),  # Shot: [0, 396]
    ("claire", "cp10_neutral", (0, 318), "claire", "zh", "cp10_neutral.wav", 0),  # Shot: [0, 318]
    ("claire", "cp11_neutral", (0, 318), "claire", "zh", "cp11_neutral.wav", 0),  # Shot: [0, 318]
    ("claire", "cp12_neutral", (0, 332), "claire", "zh", "cp12_neutral.wav", 0),  # Shot: [0, 332]
    ("claire", "cp13_neutral", (0, 295), "claire", "zh", "cp13_neutral.wav", 0),  # Shot: [0, 295]
    ("claire", "cp14_neutral", (0, 302), "claire", "zh", "cp14_neutral.wav", 0),  # Shot: [0, 302]
    ("claire", "cp15_neutral", (0, 402), "claire", "zh", "cp15_neutral.wav", 0),  # Shot: [0, 402]
    ("claire", "cp16_neutral", (0, 389), "claire", "zh", "cp16_neutral.wav", 0),  # Shot: [0, 389]
    ("claire", "cp17_neutral", (0, 372), "claire", "zh", "cp17_neutral.wav", 0),  # Shot: [0, 372]
    ("claire", "cp18_neutral", (0, 279), "claire", "zh", "cp18_neutral.wav", 0),  # Shot: [0, 279]
    ("claire", "cp19_neutral", (0, 280), "claire", "zh", "cp19_neutral.wav", 0),  # Shot: [0, 280]
    ("claire", "cp20_neutral", (0, 288), "claire", "zh", "cp20_neutral.wav", 0),  # Shot: [0, 288]
    ("claire", "cp21_neutral", (0, 550), "claire", "zh", "cp21_neutral.wav", 0),  # Shot: [0, 550]
    ("claire", "cp22_amazement", (0, 281), "claire", "zh", "cp22_amazement.wav", 0),  # Shot: [0, 281]
    ("claire", "cp23_amazement", (0, 526), "claire", "zh", "cp23_amazement.wav", 0),  # Shot: [0, 526]
    ("claire", "cp24_joy", (0, 272), "claire", "zh", "cp24_joy.wav", 0),  # Shot: [0, 272]
    ("claire", "cp25_joy", (0, 494), "claire", "zh", "cp25_joy.wav", 0),  # Shot: [0, 494]
    ("claire", "cp26_cheekiness", (0, 303), "claire", "zh", "cp26_cheekiness.wav", 0),  # Shot: [0, 303]
    ("claire", "cp27_cheekiness", (0, 574), "claire", "zh", "cp27_cheekiness.wav", 0),  # Shot: [0, 574]
    ("claire", "cp28_sadness", (0, 272), "claire", "zh", "cp28_sadness.wav", 0),  # Shot: [0, 272]
    ("claire", "cp29_sadness", (0, 710), "claire", "zh", "cp29_sadness.wav", 0),  # Shot: [0, 710]
    ("claire", "cp30_disgust", (0, 316), "claire", "zh", "cp30_disgust.wav", 0),  # Shot: [0, 316]
    ("claire", "cp31_disgust", (0, 681), "claire", "zh", "cp31_disgust.wav", 0),  # Shot: [0, 681]
    ("claire", "cp32_anger", (0, 238), "claire", "zh", "cp32_anger.wav", 0),  # Shot: [0, 238]
    ("claire", "cp33_anger", (0, 503), "claire", "zh", "cp33_anger.wav", 0),  # Shot: [0, 503]
    ("claire", "cp34_fear", (0, 286), "claire", "zh", "cp34_fear.wav", 0),  # Shot: [0, 286]
    ("claire", "cp35_fear", (0, 595), "claire", "zh", "cp35_fear.wav", 0),  # Shot: [0, 595]
    ("claire", "cp36_grief", (0, 370), "claire", "zh", "cp36_grief.wav", 0),  # Shot: [0, 370]
    ("claire", "cp37_grief", (0, 660), "claire", "zh", "cp37_grief.wav", 0),  # Shot: [0, 660]
    ("claire", "cp38_pain", (0, 296), "claire", "zh", "cp38_pain.wav", 0),  # Shot: [0, 296]
    ("claire", "cp39_pain", (0, 548), "claire", "zh", "cp39_pain.wav", 0),  # Shot: [0, 548]
    ("claire", "cp40_outofbreath", (0, 292), "claire", "zh", "cp40_outofbreath.wav", 0),  # Shot: [0, 292]
    ("claire", "cp41_outofbreath", (0, 503), "claire", "zh", "cp41_outofbreath.wav", 0),  # Shot: [0, 503]
    ("claire", "cp42_fastneutral", (0, 131), "claire", "zh", "cp42_fastneutral.wav", 0),  # Shot: [0, 131]
    ("claire", "cp43_fastneutral", (0, 270), "claire", "zh", "cp43_fastneutral.wav", 0),  # Shot: [0, 270]
    ("claire", "cp44_slowneutral", (0, 370), "claire", "zh", "cp44_slowneutral.wav", 0),  # Shot: [0, 370]
    ("claire", "cp45_slowneutral", (0, 652), "claire", "zh", "cp45_slowneutral.wav", 0),  # Shot: [0, 652]
    ("claire", "eg1_neutral", (0, 319), "claire", "en", "eg1_neutral.wav", 0),  # Shot: [0, 319]
    ("claire", "eg2_neutral", (0, 386), "claire", "en", "eg2_neutral.wav", 0),  # Shot: [0, 386]
    ("claire", "eg3_neutral", (0, 194), "claire", "en", "eg3_neutral.wav", 0),  # Shot: [0, 194]
    ("claire", "eg4_neutral", (0, 276), "claire", "en", "eg4_neutral.wav", 0),  # Shot: [0, 276]
    ("claire", "eg5_neutral", (0, 478), "claire", "en", "eg5_neutral.wav", 0),  # Shot: [0, 478]
    ("claire", "eg6_neutral", (0, 275), "claire", "en", "eg6_neutral.wav", 0),  # Shot: [0, 275]
    ("claire", "eg7_neutral", (0, 533), "claire", "en", "eg7_neutral.wav", 0),  # Shot: [0, 533]
    ("claire", "eg8_neutral", (0, 897), "claire", "en", "eg8_neutral.wav", 0),  # Shot: [0, 897]
    ("claire", "eg9_neutral", (0, 379), "claire", "en", "eg9_neutral.wav", 0),  # Shot: [0, 379]
]

########################################################################################################################
# Dataset augmentation
########################################################################################################################

# This list represents data augmentation with Muted (silent) shots/clips
# Each entry in the list corresponds to a single newly generated Muted shot
# Audio data for a Muted shot is silence
# Target animation data for a Muted shot is a still frame, copied from some source shot from the training dataset
# For each Muted shot, desired generated length is specified in frames (implying fps is same as in dataset CACHE_FPS)
# Item format: (actor_name, (source_shot_name, source_shot_frame), muted_shot_len, shot_emotion_name)
# Example: ("mark", ("shot2", 12), 120, "sadness")
AUG_MUTED_SHOTS = [
    ("claire", ("cp1_neutral", 0), 120, "neutral"),
    ("claire", ("cp23_amazement", 479), 120, "amazement"),
    ("claire", ("cp24_joy", 9), 120, "joy"),
    ("claire", ("cp26_cheekiness", 12), 120, "cheekiness"),
    ("claire", ("cp28_sadness", 0), 120, "sadness"),
    ("claire", ("cp30_disgust", 0), 120, "disgust"),
    ("claire", ("cp33_anger", 191), 120, "anger"),
    ("claire", ("cp35_fear", 22), 120, "fear"),
    ("claire", ("cp36_grief", 0), 120, "grief"),
    ("claire", ("cp38_pain", 0), 120, "pain"),
    ("claire", ("cp41_outofbreath", 485), 120, "outofbreath"),
]

########################################################################################################################
# Training params
########################################################################################################################

# Number of complete passes through the entire dataset during the training
# More epochs can help to learn the dataset better, but too many epochs can lead to overfitting
NUM_EPOCHS = 50

########################################################################################################################
# Loss weights
########################################################################################################################

# This parameter controls the forcing of the predicted lip distance to match the lip distance in the training data
# The lip distance is calculated using the skin vertices from the file at SKIN_LIP_DIST_VERTEX_LIST_FPATH
# The parameter should be greater or equal to zero (zero meaning no forcing)
LOSS_LIP_DIST_ALPHA = 1e2

# This parameter controls the forcing of the phoneme head prediction to match the phoneme in the training data
LOSS_PHONEME_ALPHA = None
LOSS_PHONEME_MOTION_ALPHA = None
