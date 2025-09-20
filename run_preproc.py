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
parser.add_argument("actor_name", type=str, help="Actor Name")
args = parser.parse_args()

validate_identifier_or_exit(args.config_name, "CONFIG_NAME")
validate_identifier_or_exit(args.actor_name, "ACTOR_NAME")

print(f"Using Config Name: {args.config_name}")
print(f"Using Actor Name: {args.actor_name}")

run_process(
    [
        os.path.join(ROOT_DIR, "docker", "run_preproc.sh"),
        args.actor_name,
        json.dumps(load_config(args.config_name, "preproc")),
        json.dumps(load_config(args.config_name, "dataset")),
    ]
)
