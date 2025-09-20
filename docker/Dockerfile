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
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

LABEL maintainer="Dmitry Korobchenko <dkorobchenko@nvidia.com>"

RUN apt-get update && apt-get install -y clang libtool-bin git python3-pip libsndfile1-dev

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN pip3 install poetry==1.8.2
COPY poetry.lock pyproject.toml /poetry/
RUN cd /poetry/ && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root
