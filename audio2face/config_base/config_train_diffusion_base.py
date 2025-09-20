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
RUN_NAME = "default"

# Additional information describing the Training run, will be saved to <TRAINING_RUN_NAME_FULL>/configs/info.txt
RUN_INFO = ""

# The version of the final trained network artifact at deployment, stored in network_info.json
NETWORK_VERSION = "0.1"

NETWORK_TYPE = "diffusion"

########################################################################################################################
# Preprocessing artifacts
########################################################################################################################

# This location is used to read input Preprocessing artifacts for Train and Deploy
PREPROC_ROOT = "/workspace/output_preproc"

# This parameter should contain the full name of the directory with Preprocessing artifacts (per actor)
# Set this parameter after running "run_preproc" and before running "run_train"
# Details about the preproc artifacts are located at <PREPROC_RUN_NAME_FULL>/configs/info.txt
# Actor-specific parameters are represented as a dictionary: PARAM = {"actor1": value1, "actor2": value2, ...}
PREPROC_RUN_NAME_FULL = {}

# Paths to specific files from preproc artifacts (per actor)
# If the value for an actor is missing -> Resolve from PREPROC_ROOT and PREPROC_RUN_NAME_FULL
SKIN_PCA_COEFFS_FPATH = {}
SKIN_PCA_SHAPES_FPATH = {}
TONGUE_PCA_COEFFS_FPATH = {}
TONGUE_PCA_SHAPES_FPATH = {}
MEAN_GEOMETRY_FPATH = {}
DATA_INFO_FPATH = {}
SHOT_LIST_FPATH = {}

########################################################################################################################
# Dataset sampling
########################################################################################################################

# Options: [skin_coeffs, tongue_coeffs, jaw, eye, skin_pose, tongue_pose]
TARGET_CHANNELS = [
    "skin_pose",
    "tongue_pose",
    "jaw",
    "eye",
]

# Frames-Per-Second rate of the output generated animation (if applicable)
# Ground-truth animation caches are resampled to this FPS for training
# Supported options: integer multiples of the original cache FPS (e.g. 30.0 -> 30.0, 60.0, 90, etc)
TARGET_FPS = 60.0

RESCALE_TARGET_CHANNEL_DATA = True

TRAIN_ON_RANDOM_SUBSEGMENT = True
SUBSEGMENT_MIN_FRAME = 30
SUBSEGMENT_MAX_FRAME = 600

# Options: [ nva2f | w2v ]
AUDIO_PREPROC_METHOD = "w2v"

AUDIO_PARAMS = {
    "buffer_len": 16000,
    "padding_left": 16000,
    "padding_right": 16000,
    "samplerate": 16000,
}

NO_STANDARDIZE_AUDIO = True

########################################################################################################################
# Dataset clips
########################################################################################################################

# This list represents the clips which will be used for the Training
# Set this parameter after running "run_preproc" and before running "run_train"
# Several clips could be associated with the same shot (e.g. with different range, augmented audio tracks, etc)
# Use <PREPROC_RUN_NAME_FULL>/deploy/clips_template.py as a template (per actor), modify if needed
# Item format: (actor_name, shot_name, (clip_first_frame, clip_last_frame), speaker_name, audio_lang, audio_fpath_rel, audio_offset)
# Example: ("mark", "shot1", (0, 100), "mark", "en", "shot1.wav", 0)
CLIPS = []

########################################################################################################################
# Dataset augmentation
########################################################################################################################

# Options: [timeshift, pitch_shift]
AUGMENT = {
    "timeshift": 0.0,
    "pitch_shift": {
        "min_semitones": -2,
        "max_semitones": 2,
        "p": 0.2,
    },
}

# This list represents data augmentation with Muted (silent) shots/clips
# Each entry in the list corresponds to a single newly generated Muted shot
# Audio data for a Muted shot is silence
# Target animation data for a Muted shot is a still frame, copied from some source shot from the training dataset
# For each Muted shot, desired generated length is specified in frames (implying fps is same as in dataset CACHE_FPS)
# Item format: (actor_name, (source_shot_name, source_shot_frame), muted_shot_len, shot_emotion_name)
# Example: ("mark", ("shot2", 12), 120, "sadness")
AUG_MUTED_SHOTS = []

AUG_MUTED_SKIN_IS_NEUTRAL_FOR_NEUTRAL_EMO = True
AUG_MUTED_TONGUE_IS_NEUTRAL = False

# Options: [ None | "gauss" | "mic" ]
AUG_MUTED_AUDIO_NOISE_TYPE = "mic"
AUG_MUTED_AUDIO_NOISE_SCALE = 0.1
AUG_MUTED_AUDIO_NOISE_LOCAL_SCALE_RANGE = (0.0, 1.0)

########################################################################################################################
# Network architecture and initializers
########################################################################################################################

NETWORK_NAME = "diffusion_rnn"
NETWORK_HYPER_PARAMS = {
    "feature_dim": 256,
    "jaw_latent_dim": 15,
    "tongue_latent_dim": 10,
    "eye_latent_dim": 4,
    "gru_feature_dim": 256,
    "num_gru_layers": 2,
    "emo_embedding_dim": 32,
    "actor_embedding_dim": 32,
    "hubert_feature_dim": 1536 // 2,
}
PRETRAINED_NET_FPATH = None

DIFFUSION_STEPS = 1000
DIFFUSION_NOISE_SCHEDULE = "cosine"
TIMESTEP_RESPACING = ""

STREAMING_CFG = {
    "60": {
        "audio_pad_left": 16000,
        "audio_pad_right": 16000,
        "window_size": 16000,
        "left_truncate": 15,
        "right_truncate": 15,
        "block_frame_size": 30,
    },
    "30": {
        "audio_pad_left": 16000,
        "audio_pad_right": 16000,
        "window_size": 16000,
        "left_truncate": 7,
        "right_truncate": 8,
        "block_frame_size": 15,
    },
}

########################################################################################################################
# Training params
########################################################################################################################

BATCH_SIZE = 1  # Only support batch size 1

# Number of complete passes through the entire dataset during the training
# More epochs can help to learn the dataset better, but too many epochs can lead to overfitting
NUM_EPOCHS = 400

LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 4

########################################################################################################################
# Loss weights
########################################################################################################################

LOSS_MOTION_ALPHA = 0.0

# This parameter controls the forcing of the predicted lip distance to match the lip distance in the training data
# The lip distance is calculated using the skin vertices from the file at SKIN_LIP_DIST_VERTEX_LIST_FPATH
# The parameter should be greater or equal to zero (zero meaning no forcing)
LOSS_LIP_DIST_ALPHA = 1e-5

LOSS_VELOCITY_ALPHA = 0.0
LOSS_EXP_SMOOTH_ALPHA = 25.0
LOSS_LIP_DIST_EXP = 10.0
LOSS_LIP_DIST_EMO_WEIGHTS = None
TONGUE_WEIGHT = 0.1
JAW_WEIGHT = 1.0

########################################################################################################################
# Misc
########################################################################################################################

TRAIN_OUTPUT_ROOT = "/workspace/output_train"
FINAL_DEPLOY_ROOT = "/workspace/output_deploy"
TORCH_CACHE_ROOT = "/workspace/torch_cache"

# This string represents emotion from SHOT_EMOTION_NAMES which should be mapped to [0, ..., 0] emotion vector (zeros)
# If set to None, every emotion from SHOT_EMOTION_NAMES will be mapped to some [0, ..., 1, ..., 0] emotion vector
SHOT_EMOTION_NAME_FOR_ALL_ZEROS = "neutral"

# This represents emotion from SHOT_EMOTION_NAMES to be used to compute default emotion vector for network_info.json
DEFAULT_SHOT_EMOTION_NAME = "neutral"

LOG_PERIOD = 10
REPRODUCIBLE = True
RNG_SEED_TORCH = 42
RNG_SEED_DATASET = 43

AUDIO_DATA_CACHING = True
PHONEME_DATA_CACHING = True
TARGET_DATA_CACHING = False
TARGET_OBJECTS_TO_CUDA = True
