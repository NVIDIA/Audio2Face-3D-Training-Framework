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
import json
import argparse
import pprint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from audio2face import utils
from audio2face.preproc import preproc

parser = argparse.ArgumentParser()
parser.add_argument("actor_name", type=str, help="Actor Name")
parser.add_argument("cfg_preproc_mod", type=str, nargs="?", default=None, help="Preproc Config Modifier")
parser.add_argument("cfg_dataset_mod", type=str, nargs="?", default=None, help="Dataset Config Modifier")
args = parser.parse_args()

utils.validate_identifier_or_raise(args.actor_name, "ACTOR_NAME")
cfg_preproc_mod = json.loads(args.cfg_preproc_mod) if args.cfg_preproc_mod is not None else None
cfg_dataset_mod = json.loads(args.cfg_dataset_mod) if args.cfg_dataset_mod is not None else None

print("===============================================================================================================")
print(f"Using Actor Name: {args.actor_name}")
print("===============================================================================================================")
print(f"Using Preproc Config Modifier:\n{pprint.pformat(cfg_preproc_mod, width=120)}")
print("===============================================================================================================")
print(f"Using Dataset Config Modifier:\n{pprint.pformat(cfg_dataset_mod, width=120)}")
print("===============================================================================================================")

result_preproc = preproc.run(args.actor_name, cfg_preproc_mod, cfg_dataset_mod)
