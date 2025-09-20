#!/bin/bash

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
WORKING_DIR="/framework"
RUN_CMD="python runners/run_preproc.py"

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
A2F_FRAMEWORK_ROOT="$(dirname "$SCRIPT_DIR}")"

source "${SCRIPT_DIR}/utils.sh"
read_env_file

GIT_SAFE_DIR_CMD="git config --global --add safe.directory /framework"
SCRIPT_ARGS_WRAPPED=$(wrap_and_escape_args "$@")
DOCKER_CMD="${GIT_SAFE_DIR_CMD} && cd ${WORKING_DIR} && ${RUN_CMD} ${SCRIPT_ARGS_WRAPPED}"
echo "${DOCKER_CMD}"
echo "==============================================================================================================="

docker run \
    --gpus all --cpus 20 \
    --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$A2F_DATASETS_ROOT":/datasets \
    -v "$A2F_WORKSPACE_ROOT":/workspace \
    -v "$A2F_FRAMEWORK_ROOT":/framework \
    -e EXTERNAL_A2F_DATASETS_ROOT="$A2F_DATASETS_ROOT" \
    -e EXTERNAL_A2F_WORKSPACE_ROOT="$A2F_WORKSPACE_ROOT" \
    -e EXTERNAL_A2F_FRAMEWORK_ROOT="$A2F_FRAMEWORK_ROOT" \
    --hostname $(compose_docker_hostname) \
    audio2face-framework-env:latest \
    /bin/bash -c "${DOCKER_CMD}"
