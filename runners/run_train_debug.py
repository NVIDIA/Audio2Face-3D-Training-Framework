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
from audio2face import train_regression, train_diffusion, utils

args = SimpleNamespace(
    config_name="example-diffusion",
)

cfg_train_mod = load_config(args.config_name, "train")
cfg_dataset_mod = load_config(args.config_name, "dataset")
cfg_inference_mod = load_config(args.config_name, "inference")
network_type = utils.get_network_type_or_raise(cfg_train_mod)

print("===============================================================================================================")
print(f"Using Train Config Modifier:\n{pprint.pformat(cfg_train_mod, width=120)}")
print("===============================================================================================================")
print(f"Using Dataset Config Modifier:\n{pprint.pformat(cfg_dataset_mod, width=120)}")
print("===============================================================================================================")
print(f"Using Inference Config Modifier:\n{pprint.pformat(cfg_inference_mod, width=120)}")
print("===============================================================================================================")

if network_type == "regression":
    result_train = train_regression.run(cfg_train_mod, cfg_dataset_mod)
elif network_type == "diffusion":
    result_train = train_diffusion.run(cfg_train_mod, cfg_dataset_mod, cfg_inference_mod)
