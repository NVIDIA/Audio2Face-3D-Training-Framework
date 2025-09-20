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
import pprint
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils import load_config
from audio2face import infer_diffusion, utils

args = SimpleNamespace(
    training_run_name_full="XXXXXX_XXXXXX_XXX",
    config_name="example-diffusion",
)

utils.validate_identifier_or_raise(args.training_run_name_full, "TRAINING_RUN_NAME_FULL")

cfg_train_mod = load_config(args.config_name, "train")
cfg_dataset_mod = load_config(args.config_name, "dataset")
cfg_inference_mod = load_config(args.config_name, "inference")

print("===============================================================================================================")
print(f"Using Training Run Name Full: {args.training_run_name_full}")
print("===============================================================================================================")
print(f"Using Train Config Modifier:\n{pprint.pformat(cfg_train_mod, width=120)}")
print("===============================================================================================================")
print(f"Using Dataset Config Modifier:\n{pprint.pformat(cfg_dataset_mod, width=120)}")
print("===============================================================================================================")
print(f"Using Inference Config Modifier:\n{pprint.pformat(cfg_inference_mod, width=120)}")
print("===============================================================================================================")

result_train = infer_diffusion.run(args.training_run_name_full, cfg_train_mod, cfg_dataset_mod, cfg_inference_mod)
