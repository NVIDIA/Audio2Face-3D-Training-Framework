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
function read_env_file() {
    SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
    ENV_FPATH="${SCRIPT_DIR}/../.env"
    if [ ! -e "$ENV_FPATH" ]; then
        echo "File $ENV_FPATH does not exist."
        exit 1
    fi
    echo "==============================================================================================================="
    cat $ENV_FPATH
    echo "==============================================================================================================="
    set -o allexport
    source $ENV_FPATH
    set +o allexport
}

function wrap_and_escape_args() {
    # Wrapping each argument with single quotes and escaping
    if [ "$#" -eq 0 ]; then
        echo ""
    else
        for arg in "$@"; do
            printf "'%s' " "$(printf "%s" "$arg" | sed "s/'/'\"'\"'/g")"
        done
    fi
}

function compose_docker_hostname() {
    echo "$(hostname)__${USER}__docker"
}
