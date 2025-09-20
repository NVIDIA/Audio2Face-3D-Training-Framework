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
import datetime
import shutil
import logging

from audio2face import utils
from audio2face.config_base import config_preproc_base, config_dataset_base
from audio2face.preproc import preproc_skin, preproc_tongue, preproc_deploy

FRAMEWORK_ROOT_DIR = utils.get_framework_root_dir()


def export_meta_data(
    actor_name: str,
    configs_dir: str,
    cfg_preproc: utils.EasyDict,
    cfg_dataset: utils.EasyDict,
    cfg_preproc_mod: dict | None = None,
    cfg_dataset_mod: dict | None = None,
) -> None:
    shutil.copy(os.path.join(FRAMEWORK_ROOT_DIR, "VERSION.md"), configs_dir)
    with open(os.path.join(configs_dir, "info.txt"), "w") as f:
        f.write(cfg_preproc.RUN_INFO)
    if cfg_preproc_mod is not None:
        utils.json_dump_pretty(cfg_preproc_mod, os.path.join(configs_dir, "config_preproc_modifier.json"))
    if cfg_dataset_mod is not None:
        utils.json_dump_pretty(cfg_dataset_mod, os.path.join(configs_dir, "config_dataset_modifier.json"))
    utils.json_dump_pretty({"actor_name": actor_name}, os.path.join(configs_dir, "actor_info.json"))

    if not utils.is_partial_exposure():
        utils.json_dump_pretty(cfg_preproc, os.path.join(configs_dir, "config_preproc_full.json"))
        utils.json_dump_pretty(cfg_dataset, os.path.join(configs_dir, "config_dataset_full.json"))
        utils.json_dump_pretty(utils.get_state_info(FRAMEWORK_ROOT_DIR), os.path.join(configs_dir, "state.json"))


def run(
    actor_name: str,
    cfg_preproc_mod: dict | None = None,
    cfg_dataset_mod: dict | None = None,
) -> dict:
    run_name = utils.get_module_var("RUN_NAME", config_preproc_base, cfg_preproc_mod)
    utils.validate_identifier_or_raise(run_name, "Preproc RUN_NAME")
    run_name_full = datetime.datetime.today().strftime("%y%m%d_%H%M%S_") + run_name

    preproc_output_root = utils.get_module_var("PREPROC_OUTPUT_ROOT", config_preproc_base, cfg_preproc_mod)
    out_dir = os.path.normpath(os.path.join(preproc_output_root, run_name_full))
    configs_dir = os.path.normpath(os.path.join(out_dir, "configs"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    utils.setup_logging(os.path.join(out_dir, "log.log"))
    logging.info("--------------------------------------------------------------------------------")
    logging.info(f"Preprocessing run: {run_name_full}")
    logging.info("--------------------------------------------------------------------------------")

    utils.validate_cfg_mod(cfg_preproc_mod, config_preproc_base, "preproc")
    utils.validate_cfg_mod(cfg_dataset_mod, config_dataset_base, "dataset")
    cfg_preproc = utils.module_to_easy_dict(config_preproc_base, modifier=cfg_preproc_mod)
    cfg_dataset = utils.module_to_easy_dict(config_dataset_base, modifier=cfg_dataset_mod)
    utils.validate_identifier_or_raise(actor_name, "Actor Name")

    export_meta_data(actor_name, configs_dir, cfg_preproc, cfg_dataset, cfg_preproc_mod, cfg_dataset_mod)

    result_skin = preproc_skin.run(actor_name, run_name_full, cfg_preproc_mod, cfg_dataset_mod)
    result_tongue = preproc_tongue.run(actor_name, run_name_full, cfg_preproc_mod, cfg_dataset_mod)
    result_deploy = preproc_deploy.run(actor_name, run_name_full, cfg_preproc_mod, cfg_dataset_mod)

    logging.info(f"Mapping to local FS: /framework is {os.getenv('EXTERNAL_A2F_FRAMEWORK_ROOT') or '/framework'}")
    logging.info(f"Mapping to local FS: /datasets is {os.getenv('EXTERNAL_A2F_DATASETS_ROOT') or '/datasets'}")
    logging.info(f"Mapping to local FS: /workspace is {os.getenv('EXTERNAL_A2F_WORKSPACE_ROOT') or '/workspace'}")
    logging.info("--------------------------------------------------------------------------------")
    logging.info(f"Preproc Run Name Full: {run_name_full}")
    logging.info("--------------------------------------------------------------------------------")

    return {
        "run_name_full": run_name_full,
        "result_skin": result_skin,
        "result_tongue": result_tongue,
        "result_deploy": result_deploy,
    }
