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
import os
import shutil
import glob
import logging
from collections import OrderedDict
import numpy as np

from audio2face import utils
from audio2face.config_base import config_preproc_base, config_dataset_base


def get_default_data(fpath, shape: tuple, dtype: np.dtype = np.float32) -> np.ndarray:
    if fpath is not None:
        return np.load(fpath)
    return np.zeros(shape, dtype)


def infer_emotion_name(cfg_dataset: dict, shot_emotion_map: dict, actor_name: str, shot_name: str) -> str:
    if (actor_name, shot_name) in shot_emotion_map.keys():
        shot_emotion_name = shot_emotion_map[(actor_name, shot_name)]
        if shot_emotion_name not in cfg_dataset.SHOT_EMOTION_NAMES:
            msg = f'Emotion name "{shot_emotion_name}" from SHOT_EMOTION_MAP list is not presented'
            msg += f" in the SHOT_EMOTION_NAMES list: {cfg_dataset.SHOT_EMOTION_NAMES}"
            raise ValueError(msg)
        return shot_emotion_name
    shot_emotion_name = "neutral"
    for emo in cfg_dataset.SHOT_EMOTION_NAMES:
        if emo in shot_name:
            shot_emotion_name = emo
            break
    return shot_emotion_name


def infer_lang(shot_name: str) -> str:
    return "en"  # TODO improve automatic language detection logic, maybe do similar to infer_emotion_name()


def compute_transform_params(reference_data: np.ndarray, transform_scale: float) -> tuple[np.ndarray, float]:
    """
    Compute the transform parameters for the reference data.
    Args:
        reference_data: The reference data to compute the transform parameters for.
        transform_scale: The scale of the output data you want to get.
    Returns:
        translate: The translate of the reference data.
        scale: The scale of the reference data.
    """
    translate = reference_data.mean(0)
    centralized = reference_data - translate
    max_value = np.abs(centralized).max()
    if max_value == 0:
        scale = 1
    else:
        scale = max_value / transform_scale
    return translate, scale


def get_transform_params(deploy_data: dict, transform_dict: dict) -> dict:
    """
    Get transform parameters from the transform method.
    Args:
        deploy_data: Dictionary with the deploy data
        transform_dict: Dictionary with the following structure:
            {
                "channel1": scale1,
                "channel2": scale2,
                ...
            }

    Returns:
        Dictionary with the transform parameters
        Eventually will be used as follows:
        rescaled = (data - final_translate) / final_scale
    """
    correspondance_dict = {
        "skin": "shapes_mean_skin",
        "tongue": "shapes_mean_tongue",
        "jaw": "neutral_jaw",
        "eye": "saccade_rot_matrix",
    }

    transform_params = {}
    for channel, scale in transform_dict.items():
        reference_data = deploy_data[correspondance_dict[channel]]
        translate, scale = compute_transform_params(reference_data, scale)

        transform_params[f"translate_{channel}"] = translate.tolist()
        transform_params[f"scale_{channel}"] = scale

    return transform_params


def compute_mean_geometry(
    skin_shapes_mean: np.ndarray,
    tongue_shapes_mean: np.ndarray,
    transform_params: dict,
    jaw_dim: int = 15,
    eye_dim: int = 4,
) -> np.ndarray:
    """
    Compute the mean geometry for diffusion models by properly scaling the geometry components.

    Args:
        skin_shapes_mean: Mean skin shape
        tongue_shapes_mean: Mean tongue shape
        transform_params: Dictionary with transform parameters (translate_skin, scale_skin, etc.)
        jaw_dim: Dimension of the jaw data (default: 15 for 5 keypoints x 3)
        eye_dim: Dimension of the eye data (default: 4 for 2 angles x 2 eyes)

    Returns:
        Combined and properly scaled mean geometry
    """
    # Get transform parameters
    translate_skin = np.array(transform_params["translate_skin"])
    scale_skin = transform_params["scale_skin"]
    translate_tongue = np.array(transform_params["translate_tongue"])
    scale_tongue = transform_params["scale_tongue"]

    # Reshape and normalize skin mean
    skin_mean_flat = skin_shapes_mean.reshape(-1, 3)
    template_face = (skin_mean_flat - translate_skin) / scale_skin

    # Reshape and normalize tongue mean
    tongue_mean_flat = tongue_shapes_mean.reshape(-1, 3)
    template_tongue = (tongue_mean_flat - translate_tongue) / scale_tongue

    # Combine all components into a single template
    # Face and tongue are rescaled, jaw and eye are left as zeros for the template
    template = np.concatenate(
        [
            template_face.reshape(-1),
            template_tongue.reshape(-1),
            np.zeros(jaw_dim + eye_dim),  # No need to include jaw and eye in the template
        ]
    )

    logging.info(f"Mean geometry template shape: {template.shape}")
    logging.info(f"Min value: {template.min()}")
    logging.info(f"Max value: {template.max()}")

    return template


def run(
    actor_name: str,
    run_name_full: str,
    cfg_preproc_mod: dict | None = None,
    cfg_dataset_mod: dict | None = None,
) -> dict:
    cfg_preproc = utils.module_to_easy_dict(config_preproc_base, modifier=cfg_preproc_mod)
    cfg_dataset = utils.module_to_easy_dict(config_dataset_base, modifier=cfg_dataset_mod)

    if run_name_full is None:
        skin_pca_out_dir = os.path.join(cfg_preproc.PREPROC_OUTPUT_ROOT, "pca", "skin")
        tongue_pca_out_dir = os.path.join(cfg_preproc.PREPROC_OUTPUT_ROOT, "pca", "tongue")
        deploy_dir = os.path.join(cfg_preproc.PREPROC_OUTPUT_ROOT, "deploy")
    else:
        utils.validate_identifier_or_raise(run_name_full, "Preproc Run Name Full")
        skin_pca_out_dir = os.path.join(cfg_preproc.PREPROC_OUTPUT_ROOT, run_name_full, "pca", "skin")
        tongue_pca_out_dir = os.path.join(cfg_preproc.PREPROC_OUTPUT_ROOT, run_name_full, "pca", "tongue")
        deploy_dir = os.path.join(cfg_preproc.PREPROC_OUTPUT_ROOT, run_name_full, "deploy")

    logging.info("--------------------------------------------------------------------------------")
    logging.info(f"Using this skin pca dir: {os.path.normpath(skin_pca_out_dir)}")
    logging.info(f"Using this tongue pca dir: {os.path.normpath(tongue_pca_out_dir)}")
    logging.info(f"Deploying to: {os.path.normpath(deploy_dir)}")
    logging.info("--------------------------------------------------------------------------------")

    os.makedirs(deploy_dir, exist_ok=True)

    skin_pca_shapes_fpath = os.path.join(skin_pca_out_dir, "skin_pca_shapes.npz")
    skin_pca_coeffs_fpath = os.path.join(skin_pca_out_dir, "skin_pca_coeffs_all.npz")
    tongue_pca_shapes_fpath = os.path.join(tongue_pca_out_dir, "tongue_pca_shapes.npz")
    jaw_keypoints_neutral_fpath = cfg_dataset.JAW_KEYPOINTS_NEUTRAL_FPATH.get(actor_name)
    eye_blink_keys_fpath = cfg_dataset.EYE_BLINK_KEYS_FPATH.get(actor_name)
    eye_saccade_rotations_fpath = cfg_dataset.EYE_SACCADE_ROTATIONS_FPATH.get(actor_name)
    skin_lip_open_pose_delta_fpath = cfg_dataset.SKIN_LIP_OPEN_POSE_DELTA_FPATH.get(actor_name)
    skin_eye_close_pose_delta_fpath = cfg_dataset.SKIN_EYE_CLOSE_POSE_DELTA_FPATH.get(actor_name)

    skin_pca_shapes = np.load(skin_pca_shapes_fpath)
    skin_pca_coeffs = np.load(skin_pca_coeffs_fpath)
    tongue_pca_shapes = np.load(tongue_pca_shapes_fpath)

    lip_open_pose_delta = get_default_data(skin_lip_open_pose_delta_fpath, skin_pca_shapes["shapes_mean"].shape)
    eye_close_pose_delta = get_default_data(skin_eye_close_pose_delta_fpath, skin_pca_shapes["shapes_mean"].shape)
    jaw_keypoints_neutral = get_default_data(jaw_keypoints_neutral_fpath, cfg_preproc.DEFAULT_JAW_KEYPOINTS_SHAPE)
    eye_blink_keys = get_default_data(eye_blink_keys_fpath, cfg_preproc.DEFAULT_EYE_BLINK_KEYS_SHAPE)
    eye_saccade_rotations = get_default_data(eye_saccade_rotations_fpath, cfg_preproc.DEFAULT_EYE_SACCADE_ROT_SHAPE)

    skin_shapes_matrix = skin_pca_shapes["shapes_matrix"]
    tongue_shapes_matrix = tongue_pca_shapes["shapes_matrix"]
    if cfg_dataset.SKIN_NEUTRAL_INFERENCE_FPATH.get(actor_name) is not None:
        skin_shapes_mean = np.load(cfg_dataset.SKIN_NEUTRAL_INFERENCE_FPATH.get(actor_name))
    else:
        skin_shapes_mean = skin_pca_shapes["shapes_mean"]
    if cfg_dataset.TONGUE_NEUTRAL_INFERENCE_FPATH.get(actor_name) is not None:
        tongue_shapes_mean = np.load(cfg_dataset.TONGUE_NEUTRAL_INFERENCE_FPATH.get(actor_name))
    else:
        tongue_shapes_mean = tongue_pca_shapes["shapes_mean"]

    #########################################################################################################

    deploy_data = {
        "shapes_matrix_skin": skin_shapes_matrix,
        "shapes_mean_skin": skin_shapes_mean,
        "neutral_skin": skin_shapes_mean,
        "lip_open_pose_delta": lip_open_pose_delta,
        "eye_close_pose_delta": eye_close_pose_delta,
        "shapes_matrix_tongue": tongue_shapes_matrix,
        "shapes_mean_tongue": tongue_shapes_mean,
        "neutral_tongue": tongue_shapes_mean,
        "neutral_jaw": jaw_keypoints_neutral,
        "blink_keys": eye_blink_keys,
        "saccade_rot_matrix": eye_saccade_rotations,
    }

    for k, v in deploy_data.items():
        logging.info(f"{k:>20} :: {v.shape} :: {v.dtype}")
    logging.info("--------------------------------------------------------------------------------")

    #########################################################################################################

    data_info = {
        "num_shapes_skin": skin_shapes_matrix.shape[0],
        "num_verts_skin": skin_shapes_matrix.shape[1],
        "num_shapes_tongue": tongue_shapes_matrix.shape[0],
        "num_verts_tongue": tongue_shapes_matrix.shape[1],
        "num_keypoints_jaw": jaw_keypoints_neutral.shape[0],
        "num_angles_eye": eye_saccade_rotations.shape[1],
    }

    # Add scale and translation parameters for diffusion support
    default_transform_scale = {
        "skin": 1,
        "tongue": 1,
        "jaw": 1,
        "eye": 1,
    }
    transform_scale = cfg_dataset.TRANSFORM_SCALE.get(actor_name, default_transform_scale)
    logging.info(f"Using transform scale: {transform_scale}")
    transform_params = get_transform_params(deploy_data, transform_scale)
    data_info.update(transform_params)

    # Compute and save mean geometry for diffusion models
    mean_geometry = compute_mean_geometry(
        skin_shapes_mean,
        tongue_shapes_mean,
        transform_params,
        jaw_keypoints_neutral.shape[0] * jaw_keypoints_neutral.shape[1],
        eye_saccade_rotations.shape[1] * 2,  # 2 eyes
    )

    mean_geometry_fpath = os.path.join(deploy_dir, "mean_geometry.npy")
    logging.info(f"Saving mean geometry to: {mean_geometry_fpath}")
    np.save(mean_geometry_fpath, mean_geometry)

    for k, v in data_info.items():
        logging.info(f"{k:>20} :: {v}")
    logging.info("--------------------------------------------------------------------------------")

    #########################################################################################################

    shot_emotion_map = {
        (_actor_name, _shot_name): _shot_emotion_name
        for (_actor_name, _shot_name, _shot_emotion_name) in cfg_dataset.SHOT_EMOTION_MAP
    }

    shot_list_json_str = "[\n"
    for i, (shot_name, coeffs) in enumerate(skin_pca_coeffs.items()):
        shot_len = coeffs.shape[0]
        shot_fps = cfg_dataset.CACHE_FPS[actor_name]
        shot_emotion_name = infer_emotion_name(cfg_dataset, shot_emotion_map, actor_name, shot_name)
        shot_list_json_str += f'    ["{actor_name}", "{shot_name}", {shot_len}, {shot_fps}, "{shot_emotion_name}"]'
        shot_list_json_str += "\n" if i == len(skin_pca_coeffs) - 1 else ",\n"
    shot_list_json_str += "]"

    clips_template_str = "CLIPS = [\n"
    for shot_name, coeffs in skin_pca_coeffs.items():
        first_frame = 0
        last_frame = coeffs.shape[0] - 1
        speaker_name = actor_name  # default speaker name is actor name
        audio_lang = infer_lang(shot_name)
        audio_fpath_rel = f"{shot_name}.wav"
        audio_offset = 0
        clips_template_str += f'    ("{actor_name}", "{shot_name}", ({first_frame}, {last_frame}), "{speaker_name}",'
        clips_template_str += f' "{audio_lang}", "{audio_fpath_rel}", {audio_offset}),'
        clips_template_str += f"  # Shot: [{first_frame}, {last_frame}]\n"

    aug_audios = OrderedDict()
    for aug_fpath in glob.glob(os.path.join(cfg_dataset.AUDIO_ROOT[actor_name], "aug", "*", "*.wav")):
        aug_speaker_dir, aug_fname = os.path.split(aug_fpath)
        _, aug_speaker = os.path.split(aug_speaker_dir)
        if aug_speaker not in aug_audios.keys():
            aug_audios[aug_speaker] = []
        aug_audios[aug_speaker].append(aug_fname)
    for aug_speaker in aug_audios.keys():
        substr = ""
        for shot_name, coeffs in skin_pca_coeffs.items():
            for aug_fname in aug_audios[aug_speaker]:
                if shot_name in aug_fname:
                    first_frame = 0
                    last_frame = coeffs.shape[0] - 1
                    speaker_name = f"aug_{aug_speaker}"
                    audio_lang = infer_lang(shot_name)
                    audio_fpath_rel = f"aug/{aug_speaker}/{aug_fname}"
                    audio_offset = 0
                    substr += f'    ("{actor_name}", "{shot_name}", ({first_frame}, {last_frame}), "{speaker_name}",'
                    substr += f' "{audio_lang}", "{audio_fpath_rel}", {audio_offset}),'
                    substr += f"  # Shot: [{first_frame}, {last_frame}]\n"
        if substr != "":
            clips_template_str += "    ####################################################################################################################\n"
            clips_template_str += substr

    clips_template_str += "]"

    #########################################################################################################
    # Copy blendshape data to deploy directory
    #########################################################################################################

    bs_skin_fpath = cfg_dataset.BLENDSHAPE_SKIN_FPATH.get(actor_name, None)
    bs_skin_config_fpath = cfg_dataset.BLENDSHAPE_SKIN_CONFIG_FPATH.get(actor_name, None)
    bs_tongue_fpath = cfg_dataset.BLENDSHAPE_TONGUE_FPATH.get(actor_name, None)
    bs_tongue_config_fpath = cfg_dataset.BLENDSHAPE_TONGUE_CONFIG_FPATH.get(actor_name, None)

    if (bs_skin_fpath is not None) and (bs_skin_config_fpath is not None):
        logging.info(f"Using blendshape skin data from: {bs_skin_fpath}")
        logging.info(f"Using blendshape skin config from: {bs_skin_config_fpath}")
        out_bs_skin_fpath = os.path.join(deploy_dir, f"bs_skin_{actor_name}.npz")
        shutil.copy(bs_skin_fpath, out_bs_skin_fpath)
        out_bs_skin_config_fpath = os.path.join(deploy_dir, f"bs_skin_config_{actor_name}.json")
        shutil.copy(bs_skin_config_fpath, out_bs_skin_config_fpath)
    else:
        logging.info("No blendshape skin data found")

    if (bs_tongue_fpath is not None) and (bs_tongue_config_fpath is not None):
        logging.info(f"Using blendshape tongue data from: {bs_tongue_fpath}")
        logging.info(f"Using blendshape tongue config from: {bs_tongue_config_fpath}")
        out_bs_tongue_fpath = os.path.join(deploy_dir, f"bs_tongue_{actor_name}.npz")
        shutil.copy(bs_tongue_fpath, out_bs_tongue_fpath)
        out_bs_tongue_config_fpath = os.path.join(deploy_dir, f"bs_tongue_config_{actor_name}.json")
        shutil.copy(bs_tongue_config_fpath, out_bs_tongue_config_fpath)
    else:
        logging.info("No blendshape tongue data found")

    #########################################################################################################

    deploy_fpath = os.path.join(deploy_dir, "network_data.npz")
    logging.info(f"Saving deploy data to: {deploy_fpath}")
    np.savez(deploy_fpath, **deploy_data)

    data_info_fpath = os.path.join(deploy_dir, "data_info.json")
    logging.info(f"Saving data info to: {data_info_fpath}")
    utils.json_dump_pretty(data_info, data_info_fpath)

    shot_list_fpath = os.path.join(deploy_dir, "shots.json")
    logging.info(f"Saving dataset shot list to: {shot_list_fpath}")
    with open(shot_list_fpath, "w") as f:
        f.write(shot_list_json_str + "\n")

    clips_template_fpath = os.path.join(deploy_dir, "clips_template.py")
    logging.info(f"Saving dataset CLIPS template to: {clips_template_fpath}")
    with open(clips_template_fpath, "w") as f:
        f.write(clips_template_str + "\n")
    print(clips_template_str)  # Printing the template without logging prefix

    logging.info("--------------------------------------------------------------------------------")

    return {
        "deploy_fpath": deploy_fpath,
        "data_info_fpath": data_info_fpath,
    }
