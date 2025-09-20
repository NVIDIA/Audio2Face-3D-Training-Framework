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
import sys
import re
import types
import importlib.util
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def validate_identifier(identifier) -> bool:
    if not isinstance(identifier, str):
        return False
    if not re.match(r"^[\w]([\w.-]*[\w])?$", identifier):
        return False
    if ".." in identifier:
        return False
    return True


def validate_identifier_or_exit(identifier, name: str) -> None:
    if not validate_identifier(identifier):
        print(f"[ERROR] Unsupported {name} format: {identifier}")
        sys.exit()


def load_module(module_fpath: str) -> types.ModuleType:
    module_name = "".join(c if c.isalpha() else "_" for c in module_fpath)
    spec = importlib.util.spec_from_file_location(module_name, module_fpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_to_dict(module: types.ModuleType) -> dict:
    return {k: getattr(module, k) for k in dir(module) if not k.startswith("_")}


def load_config(config_name: str, config_type: str, optional: bool = False) -> dict:
    config_module_fnames = {
        "dataset": "config_dataset.py",
        "preproc": "config_preproc.py",
        "train": "config_train.py",
        "inference": "config_inference.py",
    }
    config_module_fpath = os.path.join(ROOT_DIR, "configs", config_name, config_module_fnames[config_type])
    if not os.path.exists(config_module_fpath):
        if optional:
            return {}
        else:
            print(f"[ERROR] Unable to find config: {config_name} (type: {config_type})")
            print(f"[ERROR] Make sure this file exists: {config_module_fpath}")
            sys.exit()
    config_module = load_module(config_module_fpath)
    return module_to_dict(config_module)


def run_process(cmd_with_args: list) -> None:
    try:
        process = subprocess.Popen(cmd_with_args)
        process.wait()
    except KeyboardInterrupt:
        process.wait()
