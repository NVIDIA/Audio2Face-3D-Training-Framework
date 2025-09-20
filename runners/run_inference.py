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

from audio2face import infer_diffusion, utils

parser = argparse.ArgumentParser()
parser.add_argument("training_run_name_full", type=str, help="Training Run Name Full")
parser.add_argument("cfg_train_mod", type=str, nargs="?", default=None, help="Train Config Modifier")
parser.add_argument("cfg_dataset_mod", type=str, nargs="?", default=None, help="Dataset Config Modifier")
parser.add_argument("cfg_inference_mod", type=str, nargs="?", default=None, help="Inference Config Modifier")
args = parser.parse_args()

utils.validate_identifier_or_raise(args.training_run_name_full, "TRAINING_RUN_NAME_FULL")
cfg_train_mod = json.loads(args.cfg_train_mod) if args.cfg_train_mod is not None else None
cfg_dataset_mod = json.loads(args.cfg_dataset_mod) if args.cfg_dataset_mod is not None else None
cfg_inference_mod = json.loads(args.cfg_inference_mod) if args.cfg_inference_mod is not None else None
network_type = utils.get_network_type_or_raise(cfg_train_mod)

print("===============================================================================================================")
print(f"Using Training Run Name Full: {args.training_run_name_full}")
print("===============================================================================================================")
print(f"Using Train Config Modifier:\n{pprint.pformat(cfg_train_mod, width=120)}")
print("===============================================================================================================")
print(f"Using Dataset Config Modifier:\n{pprint.pformat(cfg_dataset_mod, width=120)}")
print("===============================================================================================================")
print(f"Using Inference Config Modifier:\n{pprint.pformat(cfg_inference_mod, width=120)}")
print("===============================================================================================================")

if network_type == "regression":
    raise NotImplementedError("Inference for regression networks is not implemented yet")
elif network_type == "diffusion":
    result_infer = infer_diffusion.run(args.training_run_name_full, cfg_train_mod, cfg_dataset_mod, cfg_inference_mod)
