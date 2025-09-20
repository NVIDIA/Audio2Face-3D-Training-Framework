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
PREPROC_RUN_NAME_FULL = {}

# Paths to specific files from preproc artifacts (per actor)
# If the value for an actor is missing -> Resolve from PREPROC_ROOT and PREPROC_RUN_NAME_FULL
SKIN_PCA_COEFFS_FPATH = {}
SKIN_PCA_SHAPES_FPATH = {}
TONGUE_PCA_COEFFS_FPATH = {}
TONGUE_PCA_SHAPES_FPATH = {}
DATA_INFO_FPATH = {}
SHOT_LIST_FPATH = {}

########################################################################################################################
# Dataset sampling
########################################################################################################################

# Options: [skin_coeffs, tongue_coeffs, jaw, eye, skin_pose, tongue_pose]
TARGET_CHANNELS = [
    "skin_coeffs",
    "tongue_coeffs",
    "jaw",
    "eye",
]

# Frames-Per-Second rate of the output generated animation (if applicable)
# Ground-truth animation caches are resampled to this FPS for training
# Supported options: integer multiples of the original cache FPS (e.g. 30.0 -> 30.0, 60.0, 90, etc)
TARGET_FPS = 30.0

RESCALE_TARGET_CHANNEL_DATA = False

# Options: [ nva2f | w2v ]
AUDIO_PREPROC_METHOD = "w2v"

AUDIO_PARAMS = {
    "buffer_len": 8320,
    "buffer_ofs": 8320 // 2,
    "samplerate": 16000,
}

SEQ_LEN = 2
SEQ_STRIDE = 1

REMOVE_CLOSING_EYE = True
EYE_DIST_THRESHOLD = 0.35

PHONEME_FORCING_LANGS = ["en"]
PHONEME_PROB_TEMPERATURE = 1.0

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
    "timeshift": 1.0,
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

NETWORK_NAME = "conv_w2v_autocorr"
NETWORK_HYPER_PARAMS = {
    "use_w2v_features": True,
    "use_autocorr_features": True,
    "w2v_freeze": True,
    "w2v_num_layers": 1,
    "autocorr_params": {
        "num_autocorr": 32,
        "num_windows": 25,
        "win_len": 640,
        "win_stride": 320,
        "preemph": 0.0,
        "remove_dc": True,
        "win_func": "hanning",
    },
    "implicit_emotion_len": 16,
    "explicit_emo_emb_len": 8,
}
EMO_INIT_SIGMA = 0.01
PRETRAINED_NET_FPATH = None

########################################################################################################################
# Training params
########################################################################################################################

BATCH_SIZE = 32

# Number of complete passes through the entire dataset during the training
# More epochs can help to learn the dataset better, but too many epochs can lead to overfitting
NUM_EPOCHS = 50

LEARNING_RATE = 0.001 * 0.2
EMO_LR_MULT = 1.0
LR_STEP_GAMMA = 0.994

########################################################################################################################
# Loss weights
########################################################################################################################

LOSS_MSE_ALPHA = 1.0
LOSS_MOTION_ALPHA = 10.0
LOSS_EMO_REG_ALPHA = 1.0

# This parameter controls the forcing of the predicted lip distance to match the lip distance in the training data
# The lip distance is calculated using the skin vertices from the file at SKIN_LIP_DIST_VERTEX_LIST_FPATH
# The parameter should be greater or equal to zero (zero meaning no forcing)
LOSS_LIP_DIST_ALPHA = 100.0

LOSS_LIP_DIST_EXP = 1.0
LOSS_LIP_DIST_EMO_WEIGHTS = None

LOSS_LIP_SIZE_ALPHA = None
LOSS_LIP_SIZE_EMO_WEIGHTS = None

LOSS_VOL_STAB_REG_ALPHA = 1e2
LOSS_VOL_STAB_REG_EXP = 5000.0

LOSS_PHONEME_ALPHA = 0.1
LOSS_PHONEME_SIL_WEIGHT = 1.0
LOSS_PHONEME_MOTION_ALPHA = 0.05

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

COMPACTIFY_IMPLICIT_EMO_DB = False

LOG_PERIOD = 50
REPRODUCIBLE = True
RNG_SEED_TORCH = 42
RNG_SEED_DATASET = 43

AUDIO_DATA_CACHING = True
PHONEME_DATA_CACHING = True
TARGET_DATA_CACHING = True
TARGET_OBJECTS_TO_CUDA = True
