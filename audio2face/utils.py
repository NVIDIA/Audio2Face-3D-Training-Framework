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
import glob
import re
import types
import logging
import socket
import getpass
import subprocess
import importlib
import warnings
import math
import copy
import json
from typing import TypeVar, Generic
from functools import lru_cache
import numpy as np

from audio2face.config_base.exposed import EXPOSED_CONFIG_PARAMS

EasyDictValueType = TypeVar("EasyDictValueType")

CFG_PARAM_PATH_SEP = "/"

# Properties of external components
W2V_AUDIO_SAMPLERATE = 16000  # Wav2Vec property: 16KHz input audio
W2V_FEATURE_FPS = 50  # Wav2Vec property: 50 features per second
HUBERT_FEATURE_FPS = 50  # HuBERT property: 50 features per second


class EasyDict(dict[str, EasyDictValueType], Generic[EasyDictValueType]):
    def __setattr__(self, name: str, value: EasyDictValueType) -> None:
        self[name] = value

    def __getattr__(self, name: str) -> EasyDictValueType:
        if name not in self:
            raise AttributeError(f'Dict has no attribute "{name}"')
        return self[name]

    def __delattr__(self, name: str) -> None:
        if name not in self:
            raise AttributeError(f'Dict has no attribute "{name}"')
        del self[name]


def dict_deep_update(data: dict, modifier: dict) -> dict:
    data_with_mod = copy.deepcopy(data)
    for k, v in modifier.items():
        if k in data_with_mod and isinstance(v, dict) and isinstance(data_with_mod[k], dict):
            data_with_mod[k] = dict_deep_update(data_with_mod[k], v)
        else:
            data_with_mod[k] = copy.deepcopy(v)
    return data_with_mod


def convert_to_float32(data: np.ndarray | dict) -> np.ndarray | dict:
    if isinstance(data, np.ndarray):
        if data.dtype == np.float32:
            return data
        else:
            return data.astype(np.float32)
    elif isinstance(data, dict):
        return {k: convert_to_float32(v) for k, v in data.items()}
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def module_to_easy_dict(module: types.ModuleType, modifier: dict | None = None) -> EasyDict:
    data = {}
    for k in dir(module):
        if not k.startswith("_"):
            data[k] = copy.deepcopy(getattr(module, k))
    if modifier is not None:
        data = dict_deep_update(data, modifier)
    return EasyDict(data)


def get_module_var(var_name: str, module: types.ModuleType, modifier: dict | None = None):
    if modifier is not None and var_name in modifier.keys():
        return modifier[var_name]
    if var_name in dir(module):
        return getattr(module, var_name)
    else:
        raise ValueError(f"Unable to find var: {var_name}")


def json_dumps_pretty(obj, indent: int = 4, max_len: int = 120, _current_indent: int = 0) -> str:
    s = " " * _current_indent
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return json.dumps(obj)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        inner = ",\n".join(
            " " * (_current_indent + indent)
            + f"{json.dumps(k)}: {json_dumps_pretty(v, indent, max_len, _current_indent + indent)}"
            for k, v in obj.items()
        )
        return f"{{\n{inner}\n{s}}}"
    if isinstance(obj, (list, tuple)):
        if not obj:
            return "[]"
        if all(isinstance(i, (int, float)) for i in obj):
            return f"[{', '.join(json.dumps(i) for i in obj)}]"
        if all(isinstance(i, str) for i in obj):
            inner = ",\n".join(" " * (_current_indent + indent) + json.dumps(i) for i in obj)
            return f"[\n{inner}\n{s}]"
        inner_one = ", ".join(json_dumps_pretty(i, indent, max_len, 0) for i in obj)
        one_line = f"[{inner_one}]"
        if len(one_line) <= max_len:
            return one_line
        inner = ",\n".join(
            " " * (_current_indent + indent) + json_dumps_pretty(i, indent, max_len, _current_indent + indent)
            for i in obj
        )
        return f"[\n{inner}\n{s}]"
    return json.dumps(obj)


def json_dump_pretty(obj, fpath: str, indent: int = 4) -> None:
    obj_str = json_dumps_pretty(obj, indent)
    with open(fpath, "w") as f:
        f.write(obj_str)


def log_uncaught_exceptions(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        logging.warning("Keyboard Interrupt by the user")
    else:
        exc_info = (exc_type, exc_value, exc_traceback) if not is_partial_exposure() else None
        logging.critical(f"Exception: {exc_type.__name__}: {exc_value}", exc_info=exc_info)


def setup_logging(log_fpath: str | None = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_fpath is not None:
        handlers.append(logging.FileHandler(log_fpath))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )
    sys.excepthook = log_uncaught_exceptions


@lru_cache(maxsize=None)
def warn_once(message: str) -> None:
    logging.warning(message)


def int2bytes(i: int) -> bytes:
    return np.array([i], dtype=np.dtype("<i4")).tobytes()  # "little-endian"


def bytes2int(b: bytes) -> int:
    return np.frombuffer(b, dtype=np.dtype("<i4"))[0]  # "little-endian"


def get_all_subdir_names(root_dir: str) -> list[str]:
    return sorted([os.path.split(shot_dir)[1] for shot_dir in glob.glob(os.path.join(root_dir, "*"))])


def get_cache_fpath(cache_root: str, shot_name: str) -> str:
    matched_files = []
    for ext in ["npy", "xml", "pc2"]:
        matched_files.extend(glob.glob(os.path.join(cache_root, shot_name, f"{shot_name}.{ext}")))
    if len(matched_files) == 0:
        raise RuntimeError(f'Unable to get cache file path for shot "{shot_name}" at {cache_root}')
    cache_fpath = matched_files[0]
    return cache_fpath


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged_ranges = []
    for start, end in sorted_ranges:
        if not merged_ranges or merged_ranges[-1][1] < start:
            merged_ranges.append([start, end])
        else:
            merged_ranges[-1][1] = max(merged_ranges[-1][1], end)
    merged_ranges = [tuple(rng) for rng in merged_ranges]
    return merged_ranges


def get_padded_submatrix_from_range(arr: np.ndarray, first: int, last: int) -> np.ndarray:
    if first >= 0 and last + 1 <= len(arr):
        return arr[first : last + 1]
    length = last - first + 1
    res = np.zeros((length, *arr.shape[1:]), dtype=arr.dtype)
    begin = max(0, -first)
    end = min(length, len(arr) - first)
    if begin < end:
        res[begin:end] = arr[first + begin : first + end]
    return res


def get_merged_submatrix_from_ranges(arr: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate(
        [get_padded_submatrix_from_range(arr, first, last) for (first, last) in ranges],
        axis=0,
    )


def get_w2v_seq_len_per_buffer(audio_buf_len: int) -> int:
    w2v_seq_len_per_buffer = math.floor((audio_buf_len - 80) * W2V_FEATURE_FPS / W2V_AUDIO_SAMPLERATE)
    return w2v_seq_len_per_buffer  # Number of features per input audio buffer


def change_huggingface_hub_cache_root(cache_root: str) -> None:
    import huggingface_hub.constants
    import transformers.utils.hub

    huggingface_hub.constants.HF_HUB_CACHE = os.path.join(cache_root, "huggingface/hub")
    importlib.reload(transformers.utils.hub)  # To propagate the effect of changing HF_HUB_CACHE


def suppress_transformers_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Passing `gradient_checkpointing` to a config initialization is deprecated",
        category=UserWarning,
    )


def validate_actor_name(
    actor_name: str,
    actor_names: list[str],
    msg_prefix: str | None = None,
) -> None:
    if actor_name not in actor_names:
        msg = f"{msg_prefix} : " if msg_prefix is not None else ""
        msg += f'Actor name "{actor_name}" is not presented'
        msg += f" in the ACTOR_NAMES list: {actor_names}"
        raise ValueError(msg)


def validate_shot_emotion_name(
    shot_emotion_name: str,
    shot_emotion_names: list[str],
    msg_prefix: str | None = None,
) -> None:
    if shot_emotion_name not in shot_emotion_names:
        msg = f"{msg_prefix} : " if msg_prefix is not None else ""
        msg += f'Emotion name "{shot_emotion_name}" is not presented'
        msg += f" in the SHOT_EMOTION_NAMES list: {shot_emotion_names}"
        raise ValueError(msg)


def get_network_emotion_names(shot_emotion_names: list[str], shot_emotion_name_for_all_zeros: str | None) -> list[str]:
    network_emotion_names = shot_emotion_names.copy()
    if shot_emotion_name_for_all_zeros is not None:
        validate_shot_emotion_name(
            shot_emotion_name_for_all_zeros,
            shot_emotion_names,
            msg_prefix="All zeros emo",
        )
        network_emotion_names.remove(shot_emotion_name_for_all_zeros)
    return network_emotion_names


def validate_identifier(identifier) -> bool:
    if not isinstance(identifier, str):
        return False
    if not re.match(r"^[\w]([\w.-]*[\w])?$", identifier):
        return False
    if ".." in identifier:
        return False
    return True


def validate_identifier_or_raise(identifier, name: str) -> None:
    if not validate_identifier(identifier):
        raise ValueError(f"Unsupported {name} format: {identifier}")


def get_network_type_or_raise(cfg_train_mod: dict | None) -> str:
    if cfg_train_mod is None or "NETWORK_TYPE" not in cfg_train_mod:
        raise ValueError("Missing NETWORK_TYPE in cfg_train_mod")
    network_type = cfg_train_mod["NETWORK_TYPE"]
    if network_type not in ["regression", "diffusion"]:
        raise ValueError(f"Unsupported NETWORK_TYPE: {network_type}")
    return network_type


def generate_allowed_cfg_params_from_dict(cfg_dict: dict, parent_path: str | None = None) -> list[str]:
    allowed_cfg_params = []
    for key, value in cfg_dict.items():
        full_path = key if parent_path is None else f"{parent_path}{CFG_PARAM_PATH_SEP}{key}"
        if isinstance(value, dict) and value:
            allowed_cfg_params += generate_allowed_cfg_params_from_dict(value, full_path)
        else:
            allowed_cfg_params.append(full_path)
    return allowed_cfg_params


def find_unrecognized_cfg_params(
    cfg_dict: dict, allowed_cfg_params: list[str], parent_path: str | None = None
) -> list[str]:
    unrecognized_cfg_params = []
    for key, value in cfg_dict.items():
        full_path = key if parent_path is None else f"{parent_path}{CFG_PARAM_PATH_SEP}{key}"
        allowed_exact = full_path in allowed_cfg_params
        if allowed_exact:
            continue
        allowed_child = any(param_path.startswith(full_path + CFG_PARAM_PATH_SEP) for param_path in allowed_cfg_params)
        if isinstance(value, dict):
            children = find_unrecognized_cfg_params(value, allowed_cfg_params, full_path)
        if isinstance(value, dict) and (allowed_child or children):
            unrecognized_cfg_params += children
        else:
            unrecognized_cfg_params.append(full_path)
    return unrecognized_cfg_params


def validate_cfg_mod(cfg_mod: dict | None, cfg_base: types.ModuleType, cfg_type: str) -> None:
    if cfg_mod is None:
        return
    if is_partial_exposure():
        # Validate config mod against exposed params only
        unrecognized_cfg_params = find_unrecognized_cfg_params(cfg_mod, EXPOSED_CONFIG_PARAMS[cfg_type])
    else:
        # Validate config mod against all available params
        allowed_cfg_params = generate_allowed_cfg_params_from_dict(module_to_easy_dict(cfg_base))
        unrecognized_cfg_params = find_unrecognized_cfg_params(cfg_mod, allowed_cfg_params)
    for param_path in unrecognized_cfg_params:
        logging.warning(f"Unrecognized {cfg_type} config param: {param_path}")
    if len(unrecognized_cfg_params) > 0:
        raise ValueError(f"There are unrecognized {cfg_type} config params")


def is_partial_exposure() -> bool:
    return False


def get_framework_root_dir() -> str:
    # utils.py is always at audio2face/utils.py, so framework root is ".." from here
    utils_py_path = os.path.abspath(__file__)
    return os.path.normpath(os.path.join(os.path.dirname(utils_py_path), ".."))


def get_state_info(framework_root_dir: str) -> dict:
    with open(os.path.join(framework_root_dir, "VERSION.md")) as f:
        version = f.read().strip()

    try:
        git_branch_name_cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        git_branch_name = subprocess.check_output(git_branch_name_cmd, cwd=framework_root_dir).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        git_branch_name = None

    try:
        git_commit_hash_cmd = ["git", "rev-parse", "HEAD"]
        git_commit_hash = subprocess.check_output(git_commit_hash_cmd, cwd=framework_root_dir).strip().decode()[:7]
    except subprocess.CalledProcessError:
        git_commit_hash = None

    try:
        git_status_cmd = ["git", "status", "--porcelain"]
        git_status_output = subprocess.check_output(git_status_cmd, cwd=framework_root_dir).decode("utf-8").strip()
        git_clean = len(git_status_output) == 0
    except subprocess.CalledProcessError:
        git_clean = None

    try:
        user = getpass.getuser()
    except KeyError:
        user = None

    return {
        "framework": {
            "version": version,
            "git_branch": git_branch_name,
            "git_commit": git_commit_hash,
            "git_clean": git_clean,
        },
        "machine": {
            "hostname": socket.gethostname(),
            "user": user,
        },
        "path_mapping": {
            "/framework": os.getenv("EXTERNAL_A2F_FRAMEWORK_ROOT") or "/framework",
            "/datasets": os.getenv("EXTERNAL_A2F_DATASETS_ROOT") or "/datasets",
            "/workspace": os.getenv("EXTERNAL_A2F_WORKSPACE_ROOT") or "/workspace",
        },
    }
