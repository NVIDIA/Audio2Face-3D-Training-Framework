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
import json
import argparse

from utils import validate_identifier_or_exit, load_config, run_process

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("config_name", type=str, help="Config Name")
parser.add_argument("training_run_name_full", type=str, help="Training Run Name Full")
args = parser.parse_args()

validate_identifier_or_exit(args.config_name, "CONFIG_NAME")
validate_identifier_or_exit(args.training_run_name_full, "TRAINING_RUN_NAME_FULL")

print(f"Using Config Name: {args.config_name}")
print(f"Using Training Run Name Full: {args.training_run_name_full}")

run_process(
    [
        os.path.join(ROOT_DIR, "docker", "run_deploy.sh"),
        args.training_run_name_full,
        json.dumps(load_config(args.config_name, "train")),
        json.dumps(load_config(args.config_name, "inference", optional=True)),
    ]
)
