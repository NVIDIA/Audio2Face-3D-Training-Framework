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
import json
import logging
import types

from audio2face import utils
from audio2face.config_base import (
    config_train_regression_base,
    config_train_diffusion_base,
    config_inference_regression_base,
    config_inference_diffusion_base,
)
from audio2face.emotion import ImplicitEmotionManager


def get_config_train_base(network_type: str) -> types.ModuleType:
    if network_type == "diffusion":
        return config_train_diffusion_base
    elif network_type == "regression":
        return config_train_regression_base
    else:
        raise ValueError(f"Unsupported network type: {network_type}")


def get_config_inference_base(network_type: str) -> types.ModuleType:
    if network_type == "diffusion":
        return config_inference_diffusion_base
    elif network_type == "regression":
        return config_inference_regression_base
    else:
        raise ValueError(f"Unsupported network type: {network_type}")


def create_model_json(actor_names: list[str], network_type: str, bs_dicts: dict) -> dict:
    """
    Create a dict that references all model configs and data files
    """
    model_dict = {}
    if network_type == "diffusion":
        model_dict["modelConfigPaths"] = [f"model_config_{actor_name}.json" for actor_name in actor_names]
        model_dict["modelDataPaths"] = [f"model_data_{actor_name}.npz" for actor_name in actor_names]
        model_dict["blendshapePaths"] = [bs_dicts[actor_name] for actor_name in actor_names]
    elif network_type == "regression":
        model_dict["modelConfigPath"] = "model_config.json"
        model_dict["modelDataPath"] = "model_data.npz"
        model_dict["emotionDatabasePath"] = "implicit_emo_db.npz"
        model_dict["blendshapePaths"] = bs_dicts[actor_names[0]]
    else:
        raise ValueError(f"Unsupported network type: {network_type}")

    return {"networkInfoPath": "network_info.json", "networkPath": "network.trt", **model_dict}


def create_model_config(config: dict, is_diffusion: bool, emo_db_fpath: str | None = None) -> dict:
    out_cfg = {
        "input_strength": config["input_strength"],
        "upper_face_smoothing": config["upper_face_smoothing"],
        "lower_face_smoothing": config["lower_face_smoothing"],
        "upper_face_strength": config["upper_face_strength"],
        "lower_face_strength": config["lower_face_strength"],
        "face_mask_level": config["face_mask_level"],
        "face_mask_softness": config["face_mask_softness"],
        "skin_strength": config["skin_strength"],
        "blink_strength": config["blink_strength"],
        "lower_teeth_strength": config["lower_teeth_strength"],
        "lower_teeth_height_offset": config["lower_teeth_height_offset"],
        "lower_teeth_depth_offset": config["lower_teeth_depth_offset"],
        "lip_open_offset": config["lip_open_offset"],
        "tongue_strength": config["tongue_strength"],
        "tongue_height_offset": config["tongue_height_offset"],
        "tongue_depth_offset": config["tongue_depth_offset"],
        "eyeballs_strength": config["eyeballs_strength"],
        "saccade_strength": config["saccade_strength"],
        "right_eye_rot_x_offset": config["right_eye_rot_x_offset"],
        "right_eye_rot_y_offset": config["right_eye_rot_y_offset"],
        "left_eye_rot_x_offset": config["left_eye_rot_x_offset"],
        "left_eye_rot_y_offset": config["left_eye_rot_y_offset"],
        "eyelid_open_offset": config["eyelid_open_offset"],
        "eye_saccade_seed": config["eye_saccade_seed"],
    }
    if not is_diffusion:
        out_cfg["source_shot"] = config.get("source_shot")
        out_cfg["source_frame"] = config.get("source_frame")
        if (out_cfg["source_shot"] is None or out_cfg["source_frame"] is None) and emo_db_fpath is not None:
            emo_manager = ImplicitEmotionManager()
            emo_manager.load_npz(emo_db_fpath)
            default_source_shot = list(emo_manager.emo_specs.keys())[0]
            out_cfg["source_shot"] = default_source_shot
            out_cfg["source_frame"] = 0

    return out_cfg


def run(
    training_run_name_full: str,
    cfg_train_mod: dict | None = None,
    cfg_inference_mod: dict | None = None,
) -> dict:
    utils.setup_logging()

    network_type = utils.get_network_type_or_raise(cfg_train_mod)

    current_config_train_base = get_config_train_base(network_type)
    utils.validate_cfg_mod(cfg_train_mod, current_config_train_base, "train")
    cfg_train = utils.module_to_easy_dict(current_config_train_base, modifier=cfg_train_mod)

    current_config_inference_base = get_config_inference_base(network_type)
    utils.validate_cfg_mod(cfg_inference_mod, current_config_inference_base, "inference")
    cfg_inference = utils.module_to_easy_dict(current_config_inference_base, modifier=cfg_inference_mod)

    utils.validate_identifier_or_raise(training_run_name_full, "Train Run Name Full")

    preproc_info_fpath = os.path.join(
        cfg_train.TRAIN_OUTPUT_ROOT, training_run_name_full, "configs", "preproc_info.json"
    )
    with open(preproc_info_fpath, "r") as f:
        preproc_info = json.load(f)

    preproc_run_name_fulls = preproc_info["preproc_run_name_full"]
    preproc_actor_names = list(preproc_run_name_fulls.keys())

    # Validate actors
    for actor_name in preproc_actor_names:
        utils.validate_identifier_or_raise(actor_name, "Actor Name")
        utils.validate_identifier_or_raise(preproc_run_name_fulls[actor_name], "Preproc Run Name Full")

    network_info_path = os.path.join(cfg_train.TRAIN_OUTPUT_ROOT, training_run_name_full, "deploy", "network_info.json")
    with open(network_info_path, "r") as f:
        network_info = json.load(f)

    # Set up deployment directory paths
    training_run_deploy_dir = os.path.join(cfg_train.TRAIN_OUTPUT_ROOT, training_run_name_full, "deploy")
    final_deploy_dir = os.path.join(cfg_train.FINAL_DEPLOY_ROOT, training_run_name_full)

    logging.info("--------------------------------------------------------------------------------")
    logging.info(f"Using preproc artifacts from: {preproc_run_name_fulls}")
    logging.info(f"Using training artifacts from: {training_run_deploy_dir}")
    logging.info(f"Collecting deploy artifacts at: {final_deploy_dir}")
    logging.info("--------------------------------------------------------------------------------")

    os.makedirs(final_deploy_dir, exist_ok=True)

    # Copy the basic network files from training output
    shutil.copytree(training_run_deploy_dir, final_deploy_dir, dirs_exist_ok=True)

    if "identities" in network_info["params"]:
        actor_names = network_info["params"]["identities"]
        assert sorted(actor_names) == sorted(
            preproc_actor_names
        ), "Actor names in network info do not match preproc actor names"
    else:
        raise ValueError("Network info does not contain actor names")

    logging.info(f"Deploying model with actor names: {', '.join(actor_names)}")

    # Create model config files for each actor
    bs_dicts = {}
    for actor_name in actor_names:
        if network_type == "diffusion":
            file_tag = f"_{actor_name}"
            model_config_json = {
                "config": create_model_config(cfg_inference.CONFIG, is_diffusion=True),
            }
        elif network_type == "regression":
            emo_db_fpath = os.path.join(final_deploy_dir, "implicit_emo_db.npz")
            model_config_json = {
                "config": create_model_config(cfg_inference.CONFIG, is_diffusion=False, emo_db_fpath=emo_db_fpath),
            }
            file_tag = ""
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

        utils.json_dump_pretty(model_config_json, os.path.join(final_deploy_dir, f"model_config{file_tag}.json"))

        # Copy network_data.npz as model_data_{actor_name}.npz
        preproc_run_name_full = preproc_run_name_fulls[actor_name]
        preproc_deploy_data_fpath = os.path.join(
            cfg_train.PREPROC_ROOT, preproc_run_name_full, "deploy/network_data.npz"
        )
        target_path = os.path.join(final_deploy_dir, f"model_data{file_tag}.npz")
        logging.info(f"Copying preproc data for {actor_name} from: {preproc_deploy_data_fpath} to {target_path}")
        shutil.copy(preproc_deploy_data_fpath, target_path)

        # If Blendshape data exists, copy it to the deploy directory
        preproc_deploy_dir = os.path.join(cfg_train.PREPROC_ROOT, preproc_run_name_full, "deploy")
        preproc_bs_skin_fpath = os.path.join(preproc_deploy_dir, f"bs_skin_{actor_name}.npz")
        preproc_bs_skin_config_fpath = os.path.join(preproc_deploy_dir, f"bs_skin_config_{actor_name}.json")
        preproc_bs_tongue_fpath = os.path.join(preproc_deploy_dir, f"bs_tongue_{actor_name}.npz")
        preproc_bs_tongue_config_fpath = os.path.join(preproc_deploy_dir, f"bs_tongue_config_{actor_name}.json")

        bs_dicts[actor_name] = {}

        if os.path.exists(preproc_bs_skin_fpath) and os.path.exists(preproc_bs_skin_config_fpath):
            logging.info(f"Copying bs skin data for {actor_name} from: {preproc_bs_skin_fpath} to {final_deploy_dir}")
            shutil.copy(preproc_bs_skin_fpath, final_deploy_dir)
            logging.info(
                f"Copying bs skin config for {actor_name} from: {preproc_bs_skin_config_fpath} to {final_deploy_dir}"
            )
            shutil.copy(preproc_bs_skin_config_fpath, final_deploy_dir)

            bs_dicts[actor_name]["skin"] = {
                "config": f"bs_skin_config_{actor_name}.json",
                "data": f"bs_skin_{actor_name}.npz",
            }
        else:
            logging.info(f"No bs skin data found in preproc for {actor_name}")

        if os.path.exists(preproc_bs_tongue_fpath) and os.path.exists(preproc_bs_tongue_config_fpath):
            logging.info(
                f"Copying bs tongue data for {actor_name} from: {preproc_bs_tongue_fpath} to {final_deploy_dir}"
            )
            shutil.copy(preproc_bs_tongue_fpath, final_deploy_dir)
            logging.info(
                f"Copying bs tongue config for {actor_name} from: {preproc_bs_tongue_config_fpath} to {final_deploy_dir}"
            )
            shutil.copy(preproc_bs_tongue_config_fpath, final_deploy_dir)

            bs_dicts[actor_name]["tongue"] = {
                "config": f"bs_tongue_config_{actor_name}.json",
                "data": f"bs_tongue_{actor_name}.npz",
            }
        else:
            logging.info(f"No bs tongue data found in preproc for {actor_name}")

    # Create model.json
    model_json = create_model_json(actor_names, network_type, bs_dicts)
    utils.json_dump_pretty(model_json, os.path.join(final_deploy_dir, "model.json"))

    # Create a README.md file
    is_multi_actor = len(actor_names) > 1
    multi_actor_prefix = "Multi-Actor " if is_multi_actor else ""
    model_type = "Diffusion" if network_type == "diffusion" else "Regression"
    readme_text = f"# Audio2Face {multi_actor_prefix}{model_type} Model"
    readme_path = os.path.join(final_deploy_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_text)

    logging.info(f"Mapping to local FS: /framework is {os.getenv('EXTERNAL_A2F_FRAMEWORK_ROOT') or '/framework'}")
    logging.info(f"Mapping to local FS: /datasets is {os.getenv('EXTERNAL_A2F_DATASETS_ROOT') or '/datasets'}")
    logging.info(f"Mapping to local FS: /workspace is {os.getenv('EXTERNAL_A2F_WORKSPACE_ROOT') or '/workspace'}")
    logging.info("--------------------------------------------------------------------------------")
    logging.info(f"Deploy Run Name Full: {training_run_name_full}")
    logging.info("--------------------------------------------------------------------------------")

    return {
        "final_deploy_dir": final_deploy_dir,
    }
