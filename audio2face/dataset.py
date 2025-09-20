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
import logging
from collections import OrderedDict
from abc import ABC, abstractmethod
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import audiomentations

from audio2face import audio, phoneme, utils


def gen_id(actor_name: str, object_name: str, actor_names: list[str]) -> str:
    if len(actor_names) > 1:
        return f"{actor_name}::{object_name}"
    else:
        return f"{object_name}"


class AudioProvider:
    def __init__(self, cfg_train: utils.EasyDict, cfg_dataset: utils.EasyDict) -> None:
        self.cfg_train = cfg_train
        self.cfg_dataset = cfg_dataset
        self.cache: dict[str, audio.AudioTrack] = {}
        self.dummy_track = audio.AudioTrack(data=None, samplerate=self.cfg_train.AUDIO_PARAMS["samplerate"])

    def read_data(self, actor_name: str, audio_fpath_rel: str) -> audio.AudioTrack:
        audio_fpath = os.path.join(self.cfg_dataset.AUDIO_ROOT[actor_name], audio_fpath_rel)
        logging.info(f"[{actor_name}] Reading audio from: {audio_fpath}")
        return audio.read_and_preproc_audio_track(
            audio_fpath,
            preproc_method=self.cfg_train.AUDIO_PREPROC_METHOD,
            new_samplerate=self.cfg_train.AUDIO_PARAMS["samplerate"],
        )

    def get_data(self, actor_name: str, audio_fpath_rel: str) -> audio.AudioTrack:
        if not self.cfg_train.AUDIO_DATA_CACHING:
            return self.read_data(actor_name, audio_fpath_rel)
        audio_id = gen_id(actor_name, audio_fpath_rel, self.cfg_dataset.ACTOR_NAMES)
        if audio_id not in self.cache.keys():
            self.cache[audio_id] = self.read_data(actor_name, audio_fpath_rel)
        return self.cache[audio_id]

    def get_frame_data(self, audio_track: audio.AudioTrack, t: float, t_shift: float = 0.0) -> np.ndarray:
        audio_sample = audio_track.sec_to_sample(t + t_shift)
        ofs = audio_sample - self.cfg_train.AUDIO_PARAMS["buffer_ofs"]
        return audio_track.get_padded_buffer(ofs, self.cfg_train.AUDIO_PARAMS["buffer_len"])

    def get_segment_data(
        self, audio_track: audio.AudioTrack, t_range: list[float], t_shift: float = 0.0, transform=None
    ) -> np.ndarray:
        audio_sample_start = audio_track.sec_to_sample(t_range[0] + t_shift)
        audio_sample_end = audio_track.sec_to_sample(t_range[1] + t_shift)
        audio_segment_len = audio_sample_end - audio_sample_start + 1
        audio_segment_data = audio_track.get_padded_buffer(audio_sample_start, audio_segment_len)
        if transform is not None:
            audio_segment_data = transform(audio_segment_data, audio_track.samplerate)
        return audio_segment_data

    def get_muted_frame_data(self, rng: np.random.Generator | None = None) -> np.ndarray:
        return audio.generate_audio_noise(
            self.cfg_train.AUDIO_PARAMS["buffer_len"],
            self.cfg_train.AUDIO_PARAMS["samplerate"],
            self.cfg_train.AUG_MUTED_AUDIO_NOISE_TYPE,
            self.cfg_train.AUG_MUTED_AUDIO_NOISE_SCALE,
            rng,
        )

    def get_muted_seq_data(self, t_seq: list[float], rng: np.random.Generator | None = None) -> np.ndarray:
        if self.cfg_train.AUG_MUTED_AUDIO_NOISE_TYPE is None:
            return np.zeros((len(t_seq), self.cfg_train.AUDIO_PARAMS["buffer_len"]), dtype=np.float32)
        if self.cfg_train.AUG_MUTED_AUDIO_NOISE_LOCAL_SCALE_RANGE is None:
            local_noise_scale = 1.0
        else:
            local_noise_scale = rng.uniform(*self.cfg_train.AUG_MUTED_AUDIO_NOISE_LOCAL_SCALE_RANGE)
        audio_seq = [self.get_muted_frame_data(rng) * local_noise_scale for t in t_seq]
        return np.stack(audio_seq).astype(np.float32)

    def get_muted_segment_data(self, t_range: list[float], rng: np.random.Generator | None = None) -> np.ndarray:
        audio_sample_start = self.dummy_track.sec_to_sample(t_range[0])
        audio_sample_end = self.dummy_track.sec_to_sample(t_range[1])
        audio_segment_len = audio_sample_end - audio_sample_start + 1
        if self.cfg_train.AUG_MUTED_AUDIO_NOISE_TYPE is None:
            return np.zeros(audio_segment_len, dtype=np.float32)
        if self.cfg_train.AUG_MUTED_AUDIO_NOISE_LOCAL_SCALE_RANGE is None:
            local_noise_scale = 1.0
        else:
            local_noise_scale = rng.uniform(*self.cfg_train.AUG_MUTED_AUDIO_NOISE_LOCAL_SCALE_RANGE)
        audio_segment = audio.generate_audio_noise(
            audio_segment_len,
            self.cfg_train.AUDIO_PARAMS["samplerate"],
            self.cfg_train.AUG_MUTED_AUDIO_NOISE_TYPE,
            self.cfg_train.AUG_MUTED_AUDIO_NOISE_SCALE * local_noise_scale,
            rng,
        )
        return audio_segment.astype(np.float32)


class PhonemeProvider:
    def __init__(self, cfg_train: utils.EasyDict, cfg_dataset: utils.EasyDict) -> None:
        self.cfg_train = cfg_train
        self.cfg_dataset = cfg_dataset
        self.cache: dict[str, phoneme.Phonemes] = {}
        self.phoneme_detector = phoneme.PhonemeDetector(
            cfg_train.PHONEME_FORCING_LANGS, cfg_train.PHONEME_PROB_TEMPERATURE, cfg_train.TORCH_CACHE_ROOT
        )
        self.phoneme_buf_len = utils.get_w2v_seq_len_per_buffer(self.cfg_train.AUDIO_PARAMS["buffer_len"])
        self.muted_lang = self.phoneme_detector.langs[0]  # Any lang with [SIL] could be used

    def gen_data(self, actor_name: str, audio_fpath_rel: str, audio_lang: str) -> phoneme.Phonemes:
        audio_fpath = os.path.join(self.cfg_dataset.AUDIO_ROOT[actor_name], audio_fpath_rel)
        logging.info(f"[{actor_name}] Generating [{audio_lang}] phonemes from: {audio_fpath}")
        return self.phoneme_detector.gen_phonemes(audio_fpath, audio_lang, new_samplerate=utils.W2V_FEATURE_FPS)

    def get_data(self, actor_name: str, audio_fpath_rel: str, audio_lang: str) -> phoneme.Phonemes:
        if not self.cfg_train.PHONEME_DATA_CACHING:
            return self.gen_data(actor_name, audio_fpath_rel, audio_lang)
        audio_id = gen_id(actor_name, audio_fpath_rel, self.cfg_dataset.ACTOR_NAMES)
        if audio_id not in self.cache.keys():
            self.cache[audio_id] = self.gen_data(actor_name, audio_fpath_rel, audio_lang)
        return self.cache[audio_id]

    def get_frame_data(self, phonemes: phoneme.Phonemes, t: float, t_shift: float = 0.0) -> np.ndarray:
        phoneme_sample = phonemes.sec_to_sample(t + t_shift)
        ofs = phoneme_sample - self.phoneme_buf_len // 2
        sil_token_idx = self.phoneme_detector.get_sil_token_idx(phonemes.lang)
        phoneme_frame_data = phonemes.get_padded_buffer(ofs, self.phoneme_buf_len, sil_token_idx)
        return self.pad_frame_data_to_max_num_phonemes(phoneme_frame_data)

    def get_muted_frame_data(self) -> np.ndarray:
        num_phonemes = self.phoneme_detector.num_phonemes[self.muted_lang]
        sil_token_idx = self.phoneme_detector.get_sil_token_idx(self.muted_lang)
        phoneme_frame_data = np.zeros((self.phoneme_buf_len, num_phonemes), dtype=np.float32)
        phoneme_frame_data[..., sil_token_idx] = 1.0
        return self.pad_frame_data_to_max_num_phonemes(phoneme_frame_data)

    def get_dummy_frame_data(self) -> np.ndarray:
        num_phonemes = self.phoneme_detector.max_num_phonemes
        return np.zeros((self.phoneme_buf_len, num_phonemes), dtype=np.float32)

    def pad_frame_data_to_max_num_phonemes(self, phoneme_frame_data: np.ndarray) -> np.ndarray:
        padding = [(0, 0), (0, self.phoneme_detector.max_num_phonemes - phoneme_frame_data.shape[-1])]
        return np.pad(phoneme_frame_data, padding, mode="constant", constant_values=0)


class TargetProviderBase(ABC):
    def __init__(self, cfg_train: utils.EasyDict, cfg_dataset: utils.EasyDict) -> None:
        self.cfg_train = cfg_train
        self.cfg_dataset = cfg_dataset
        self.cache: dict[str, torch.Tensor] = {}
        self.data_info: dict[str, dict] = self.read_data_info()
        self.shot_list: list = self.read_shot_list()
        self.channel_size: dict[str, int] = {}
        self.channel_ofs: dict[str, int] = {}
        self.data_objects: dict[str, dict[str]] = {}
        self.calc_channel_dims()

    def object_to_tensors(self, data):
        if isinstance(data, dict):
            return {key: self.object_to_tensors(value) for key, value in data.items()}
        if isinstance(data, (np.ndarray, float, list)):
            data = torch.as_tensor(data, dtype=torch.float32)
            if self.cfg_train.TARGET_OBJECTS_TO_CUDA:
                data = data.cuda()
            return data
        return data

    def get_data_object(self, object_name: str, object_fpath: str, actor_name: str):
        if object_name not in self.data_objects.keys():
            self.data_objects[object_name] = {}
        if actor_name not in self.data_objects[object_name].keys():
            logging.info(f"[{actor_name}] Reading {object_name} from: {object_fpath}")
            data = np.load(object_fpath)
            data = dict(data) if isinstance(data, np.lib.npyio.NpzFile) else data
            self.data_objects[object_name][actor_name] = self.object_to_tensors(data)
        return self.data_objects[object_name][actor_name]

    def get_preproc_artifact_fpath(self, actor_name: str, rel_fpath: str, full_fpath: dict[str, str] = None) -> str:
        if full_fpath is not None and full_fpath.get(actor_name) is not None:
            return full_fpath.get(actor_name)
        elif self.cfg_train.PREPROC_RUN_NAME_FULL.get(actor_name) is not None:
            return os.path.join(
                self.cfg_train.PREPROC_ROOT, self.cfg_train.PREPROC_RUN_NAME_FULL.get(actor_name), rel_fpath
            )
        else:
            raise ValueError(f'PREPROC_RUN_NAME_FULL is not provided for actor "{actor_name}" in the config')

    def get_data_info_fpath(self, actor_name: str) -> str:
        return self.get_preproc_artifact_fpath(actor_name, "deploy/data_info.json", self.cfg_train.DATA_INFO_FPATH)

    def get_shot_list_fpath(self, actor_name: str) -> str:
        return self.get_preproc_artifact_fpath(actor_name, "deploy/shots.json", self.cfg_train.SHOT_LIST_FPATH)

    def read_data_info(self) -> dict[str, dict]:
        data_info = {}
        for actor_name in self.cfg_dataset.ACTOR_NAMES:
            data_info_fpath = self.get_data_info_fpath(actor_name)
            logging.info(f"[{actor_name}] Reading data info from: {data_info_fpath}")
            with open(data_info_fpath, "r") as f:
                data_info[actor_name] = json.load(f)
        return data_info

    def read_shot_list(self) -> list:
        shot_list = []
        for actor_name in self.cfg_dataset.ACTOR_NAMES:
            shot_list_fpath = self.get_shot_list_fpath(actor_name)
            logging.info(f"[{actor_name}] Reading shot list from: {shot_list_fpath}")
            with open(shot_list_fpath, "r") as f:
                shot_list += json.load(f)
        return shot_list

    def get_mean_geometry(self, actor_name: str) -> torch.Tensor:
        mean_geometry_fpath = self.get_preproc_artifact_fpath(
            actor_name, "deploy/mean_geometry.npy", self.cfg_train.MEAN_GEOMETRY_FPATH
        )
        return self.get_data_object("mean geometry", mean_geometry_fpath, actor_name)

    def get_data_info_value(self, actor_name: str, param: str):
        object_name = f"data_info_{param}"
        if object_name not in self.data_objects.keys():
            self.data_objects[object_name] = {}
        if actor_name not in self.data_objects[object_name].keys():
            value = self.data_info[actor_name][param]
            self.data_objects[object_name][actor_name] = self.object_to_tensors(value)
        return self.data_objects[object_name][actor_name]

    def get_unique_data_info_value(self, param: str):
        actor_names = list(self.data_info.keys())
        ref_actor_name = actor_names[0]
        ref_value = self.data_info[ref_actor_name][param]
        for actor_name in actor_names[1:]:
            if self.data_info[actor_name][param] != ref_value:
                msg = f'Inconsistent "{param}" in data_info: '
                msg += f'{ref_value} for actor "{ref_actor_name}", '
                msg += f'{self.data_info[actor_name][param]} for actor "{actor_name}"'
                raise ValueError(msg)
        return ref_value

    @abstractmethod
    def get_channel_size(self, channel: str) -> int:
        pass

    def calc_channel_dims(self) -> None:
        self.channel_size = {}
        self.channel_ofs = {}
        ofs = 0
        for channel in self.cfg_train.TARGET_CHANNELS:
            self.channel_size[channel] = self.get_channel_size(channel)
            self.channel_ofs[channel] = ofs
            ofs += self.channel_size[channel]

    @abstractmethod
    def get_channel_data(self, actor_name: str, shot_name: str, channel: str) -> torch.Tensor:
        pass

    def validate_channel_size(self, target_data_channel_size: int, channel: str, msg_prefix: str | None = None) -> None:
        if target_data_channel_size != self.channel_size[channel]:
            msg = f"{msg_prefix} : " if msg_prefix is not None else ""
            msg += f'Channel size mismatch for "{channel}" '
            msg += f"(expected: {self.channel_size[channel]}, actual: {target_data_channel_size})"
            raise ValueError(msg)

    def trim_channel_data(self, target_data_channels: list[torch.Tensor], msg_prefix: str | None = None) -> None:
        # This is done to avoid trimming placeholder data
        channel_len = [channel_data.shape[0] for channel_data in target_data_channels if channel_data.shape[0] > 1]
        if len(channel_len) == 0:
            raise ValueError("All the channels are placeholders or have length 1")
        common_len = min(channel_len)
        for channel_idx, channel in enumerate(self.cfg_train.TARGET_CHANNELS):
            channel_data_len = target_data_channels[channel_idx].shape[0]
            if channel_data_len > common_len:
                msg = f"{msg_prefix} : " if msg_prefix is not None else ""
                msg += f'Trimming "{channel}" data len: {channel_data_len} -> {common_len}'
                utils.warn_once(msg)
                target_data_channels[channel_idx] = target_data_channels[channel_idx][:common_len, ...]
            elif channel_data_len == 1:
                target_data_channels[channel_idx] = target_data_channels[channel_idx].repeat(common_len, 1)

    def read_data(self, actor_name: str, shot_name: str) -> torch.Tensor:
        shot_id = gen_id(actor_name, shot_name, self.cfg_dataset.ACTOR_NAMES)
        target_data_channels = []
        for channel in self.cfg_train.TARGET_CHANNELS:
            channel_data = self.get_channel_data(actor_name, shot_name, channel)
            channel_data = channel_data.reshape(channel_data.shape[0], -1)
            self.validate_channel_size(channel_data.shape[1], channel, msg_prefix=shot_id)
            target_data_channels.append(channel_data)
        self.trim_channel_data(target_data_channels, msg_prefix=shot_id)
        target_data = torch.cat(target_data_channels, dim=1)
        return target_data

    def get_data(self, actor_name: str, shot_name: str) -> torch.Tensor:
        if not self.cfg_train.TARGET_DATA_CACHING:
            return self.read_data(actor_name, shot_name)
        shot_id = gen_id(actor_name, shot_name, self.cfg_dataset.ACTOR_NAMES)
        if shot_id not in self.cache.keys():
            self.cache[shot_id] = self.read_data(actor_name, shot_name)
        return self.cache[shot_id]

    def get_frame_data(self, target_data: torch.Tensor, frame_idx: int, frame_shift: float = 0.0) -> torch.Tensor:
        if frame_shift == 0.0:
            # TODO This will fail if frame_idx is out of range
            return target_data[frame_idx, ...].clone()
        else:  # Interpolation
            extent = 1
            neighbors = []
            for e in range(-extent, extent + 1):  # [-extent, +extent]
                n_idx = np.clip(frame_idx + e, 0, len(target_data) - 1)
                neighbors.append(target_data[n_idx, ...])
            if frame_shift >= 0.0:
                (x, j) = (min(frame_shift, 1.0), 1)
            else:
                (x, j) = (max(frame_shift + 1.0, 0.0), 0)
            target_frame_data = neighbors[j + 0] * (1.0 - x)
            target_frame_data += neighbors[j + 1] * (x)
            return target_frame_data

    def get_segment_data(
        self, target_data: torch.Tensor, shot_fps: float, frame_range: list[int], frame_shift: float = 0.0
    ) -> torch.Tensor:
        if frame_shift == 0.0:
            # TODO This will fail if frame_range is out of range
            target_segment_data = target_data[frame_range[0] : frame_range[1] + 1, ...]
        else:  # Interpolation
            # TODO Implement timeshift augmentation for segment data
            raise NotImplementedError("TargetProvider: timeshift augmentation is not supported for segment data")
        target_segment_data = self.resample_target_data(target_segment_data, shot_fps, self.cfg_train.TARGET_FPS)
        return target_segment_data

    @abstractmethod
    def get_muted_frame_data(self, target_data: torch.Tensor, source_frame: int, emotion_name: str) -> torch.Tensor:
        pass

    def get_muted_segment_data(
        self, muted_frame_data: torch.Tensor, shot_fps: float, frame_range: list[int]
    ) -> torch.Tensor:
        resampling_factor = self.get_resampling_factor(self.cfg_train.TARGET_FPS, shot_fps)
        target_segment_len = (frame_range[1] - frame_range[0] + 1) * resampling_factor
        return muted_frame_data.unsqueeze(0).repeat(target_segment_len, 1).float()

    def set_frame_channel(self, target_frame_data: torch.Tensor, channel: str, value: torch.Tensor | float):
        target_frame_data[self.channel_ofs[channel] : self.channel_ofs[channel] + self.channel_size[channel]] = value

    def resample_target_data(self, target_data: torch.Tensor, shot_fps: float, target_fps: float) -> torch.Tensor:
        resampling_factor = self.get_resampling_factor(target_fps, shot_fps)
        if resampling_factor == 1.0:
            return target_data
        resampled_data_len = len(target_data) * resampling_factor
        target_data = target_data.transpose(0, 1).unsqueeze(0)
        target_data = F.interpolate(target_data, size=resampled_data_len, mode="linear", align_corners=False)
        target_data = target_data.squeeze(0).transpose(0, 1)
        return target_data

    def get_resampling_factor(self, target_fps: float, shot_fps: float) -> float:
        resampling_factor_true = target_fps / shot_fps
        resampling_factor = round(resampling_factor_true)  # TODO Support non-integer factor
        if resampling_factor_true != resampling_factor:
            msg = f"Resampling [fps={shot_fps}] to [fps={shot_fps * resampling_factor}] instead of [fps={target_fps}]. "
            msg += "Only integer resampling factor is supported."
            utils.warn_once(msg)
        return resampling_factor


class TargetProviderFullFace(TargetProviderBase):
    def __init__(self, cfg_train: utils.EasyDict, cfg_dataset: utils.EasyDict) -> None:
        super().__init__(cfg_train, cfg_dataset)

    def create_placeholder(self, channel: str) -> torch.Tensor:
        """
        We create a placeholder for the channel data. This is used when the channel data is not available.
        We create a placeholder with shape (1, channel_size) so we can detect it later during trimming.
        """
        placeholder = torch.zeros((1, self.channel_size[channel]), dtype=torch.float32)
        if self.cfg_train.TARGET_OBJECTS_TO_CUDA:
            placeholder = placeholder.cuda()
        return placeholder

    def get_channel_size(self, channel: str) -> int:
        if channel == "skin_coeffs":
            return self.get_unique_data_info_value("num_shapes_skin")
        elif channel == "tongue_coeffs":
            return self.get_unique_data_info_value("num_shapes_tongue")
        elif channel == "jaw":
            return self.get_unique_data_info_value("num_keypoints_jaw") * 3
        elif channel == "eye":
            return self.get_unique_data_info_value("num_angles_eye") * 2
        elif channel == "skin_pose":
            return self.get_unique_data_info_value("num_verts_skin") * 3
        elif channel == "tongue_pose":
            return self.get_unique_data_info_value("num_verts_tongue") * 3
        else:
            raise ValueError(f'Unsupported channel: "{channel}"')

    def get_channel_data(self, actor_name: str, shot_name: str, channel: str) -> torch.Tensor:
        if channel == "skin_coeffs":
            return self.get_skin_pca_coeffs(actor_name)[shot_name]
        elif channel == "tongue_coeffs":
            tongue_pca_coeffs = self.get_tongue_pca_coeffs(actor_name).get(shot_name, None)
            if tongue_pca_coeffs is None:
                return self.create_placeholder("tongue_coeffs")

            return tongue_pca_coeffs
        elif channel == "jaw":
            if actor_name not in self.cfg_dataset.JAW_ANIM_DATA_FPATH:
                return self.create_placeholder("jaw")
            jaw_anim_data = self.get_jaw_anim_data(actor_name)[shot_name]
            if self.cfg_train.RESCALE_TARGET_CHANNEL_DATA:
                translate_jaw = 0.0
                scale_jaw: torch.Tensor | float = self.get_data_info_value(actor_name, "scale_jaw")
                jaw_anim_data = self.rescale_anim_data(jaw_anim_data, translate_jaw, scale_jaw)
            return jaw_anim_data
        elif channel == "eye":
            if actor_name not in self.cfg_dataset.EYE_ANIM_DATA_FPATH:
                return self.create_placeholder("eye")
            eye_anim_data = self.get_eye_anim_data(actor_name)[shot_name]
            if self.cfg_train.RESCALE_TARGET_CHANNEL_DATA:
                translate_eye = 0.0
                scale_eye: torch.Tensor | float = self.get_data_info_value(actor_name, "scale_eye")
                eye_anim_data = self.rescale_anim_data(eye_anim_data, translate_eye, scale_eye)
            return eye_anim_data
        elif channel == "skin_pose":
            skin_pca_coeffs = self.get_skin_pca_coeffs(actor_name)[shot_name]
            skin_pca_shapes = self.get_skin_pca_shapes(actor_name)
            skin_pose = self.coeffs_to_pose(
                skin_pca_coeffs, skin_pca_shapes["shapes_matrix"], skin_pca_shapes["shapes_mean"]
            )
            if self.cfg_train.RESCALE_TARGET_CHANNEL_DATA:
                translate_skin: torch.Tensor | float = self.get_data_info_value(actor_name, "translate_skin")
                scale_skin: torch.Tensor | float = self.get_data_info_value(actor_name, "scale_skin")
                skin_pose = self.rescale_anim_data(skin_pose, translate_skin, scale_skin)
            return skin_pose
        elif channel == "tongue_pose":
            tongue_pca_coeffs = self.get_tongue_pca_coeffs(actor_name).get(shot_name, None)
            if tongue_pca_coeffs is None:
                return self.create_placeholder("tongue_pose")
            tongue_pca_shapes = self.get_tongue_pca_shapes(actor_name)
            tongue_pose = self.coeffs_to_pose(
                tongue_pca_coeffs, tongue_pca_shapes["shapes_matrix"], tongue_pca_shapes["shapes_mean"]
            )
            if self.cfg_train.RESCALE_TARGET_CHANNEL_DATA:
                translate_tongue: torch.Tensor | float = self.get_data_info_value(actor_name, "translate_tongue")
                scale_tongue: torch.Tensor | float = self.get_data_info_value(actor_name, "scale_tongue")
                tongue_pose = self.rescale_anim_data(tongue_pose, translate_tongue, scale_tongue)
            return tongue_pose
        else:
            raise ValueError(f'Unsupported channel: "{channel}"')

    def get_skin_pca_coeffs(self, actor_name: str) -> dict[str, torch.Tensor]:
        skin_pca_coeffs_fptath = self.get_preproc_artifact_fpath(
            actor_name, "pca/skin/skin_pca_coeffs_all.npz", self.cfg_train.SKIN_PCA_COEFFS_FPATH
        )
        return self.get_data_object("skin PCA coeffs", skin_pca_coeffs_fptath, actor_name)

    def get_skin_pca_shapes(self, actor_name: str) -> dict[str, torch.Tensor]:
        skin_pca_shapes_fpath = self.get_preproc_artifact_fpath(
            actor_name, "pca/skin/skin_pca_shapes.npz", self.cfg_train.SKIN_PCA_SHAPES_FPATH
        )
        return self.get_data_object("skin PCA shapes", skin_pca_shapes_fpath, actor_name)

    def get_tongue_pca_coeffs(self, actor_name: str) -> dict[str, torch.Tensor]:
        tongue_pca_coeffs_fpath = self.get_preproc_artifact_fpath(
            actor_name, "pca/tongue/tongue_pca_coeffs_all.npz", self.cfg_train.TONGUE_PCA_COEFFS_FPATH
        )
        return self.get_data_object("tongue PCA coeffs", tongue_pca_coeffs_fpath, actor_name)

    def get_tongue_pca_shapes(self, actor_name: str) -> dict[str, torch.Tensor]:
        tongue_pca_shapes_fpath = self.get_preproc_artifact_fpath(
            actor_name, "pca/tongue/tongue_pca_shapes.npz", self.cfg_train.TONGUE_PCA_SHAPES_FPATH
        )
        return self.get_data_object("tongue PCA shapes", tongue_pca_shapes_fpath, actor_name)

    def get_jaw_anim_data(self, actor_name: str) -> dict[str, torch.Tensor]:
        jaw_anim_data_fpath = self.cfg_dataset.JAW_ANIM_DATA_FPATH[actor_name]
        return self.get_data_object("jaw anim data", jaw_anim_data_fpath, actor_name)

    def get_eye_anim_data(self, actor_name: str) -> dict[str, torch.Tensor]:
        eye_anim_data_fpath = self.cfg_dataset.EYE_ANIM_DATA_FPATH[actor_name]
        return self.get_data_object("eye anim data", eye_anim_data_fpath, actor_name)

    def coeffs_to_pose(
        self, coeffs: torch.Tensor, shapes_matrix: torch.Tensor, shapes_mean: torch.Tensor
    ) -> torch.Tensor:
        pose = torch.matmul(coeffs, shapes_matrix.view(shapes_matrix.shape[0], -1))
        pose = pose.view(pose.shape[:-1] + (-1, 3))
        return pose + shapes_mean

    def rescale_anim_data(
        self, anim_data: torch.Tensor, translate: torch.Tensor | float, scale: torch.Tensor | float
    ) -> torch.Tensor:
        return (anim_data - translate) / scale

    def get_muted_frame_data(self, target_data: torch.Tensor, source_frame: int, emotion_name: str) -> torch.Tensor:
        target_frame_data = self.get_frame_data(target_data, source_frame)
        if self.cfg_train.AUG_MUTED_SKIN_IS_NEUTRAL_FOR_NEUTRAL_EMO and emotion_name == "neutral":
            if "skin_coeffs" in self.cfg_train.TARGET_CHANNELS:
                self.set_frame_channel(target_frame_data, "skin_coeffs", 0.0)
        if self.cfg_train.AUG_MUTED_TONGUE_IS_NEUTRAL:
            if "tongue_coeffs" in self.cfg_train.TARGET_CHANNELS:
                self.set_frame_channel(target_frame_data, "tongue_coeffs", 0.0)
        return target_frame_data


class Shot:
    def __init__(
        self,
        shot_id: str,
        actor_name: str,
        shot_name: str,
        shot_len: int,
        shot_fps: float,
        shot_emotion_name: str,
        shot_start_global: int,
        target_provider: TargetProviderBase,
    ):
        self.id = shot_id
        self.actor_name = actor_name
        self.name = shot_name
        self.len = shot_len
        self.fps = shot_fps
        self.emotion_name = shot_emotion_name
        self.start_global = shot_start_global  # start frame within all shots concatenated
        self.target_provider = target_provider

    def __repr__(self):
        return f"[{self.id}] | len = {self.len} | fps = {self.fps}"

    def frame_to_time(self, frame_idx: int) -> float:
        return (float(frame_idx) + 0.5) / self.fps

    def frame_range_to_time_range(self, frame_range: list[int]) -> list[float]:
        return [float(frame_range[0]) / self.fps, float(frame_range[1] + 1) / self.fps]

    def get_t_seq(self, frame_idx_seq: list[int]) -> list[float]:
        return [self.frame_to_time(frame_idx) for frame_idx in frame_idx_seq]

    def get_global_frame_idx_seq(self, frame_idx_seq: list[int]) -> np.ndarray:
        if max(frame_idx_seq) > self.len - 1:
            msg = f"{self.id} : Frame seq {frame_idx_seq} is out of shot range [0, {self.len - 1}]. "
            msg += "Global frame idx seq will be out of range"
            utils.warn_once(msg)
        global_frame_idx_seq = [(self.start_global + frame_idx) for frame_idx in frame_idx_seq]
        return np.stack(global_frame_idx_seq).astype(np.int64)  # idxs within all shots concatenated

    def verify_target_data(self, target_data: torch.Tensor):
        if len(target_data) != self.len:
            utils.warn_once(f"{self.id} : Shot len mismatch (expected: {self.len}, actual: {len(target_data)})")

    def get_target_seq(self, frame_idx_seq: list[int], frame_shift_aug: float = 0.0) -> torch.Tensor:
        target_data = self.target_provider.get_data(self.actor_name, self.name)
        self.verify_target_data(target_data)
        target_seq = [
            self.target_provider.get_frame_data(target_data, frame_idx, frame_shift_aug) for frame_idx in frame_idx_seq
        ]
        return torch.stack(target_seq).float()

    def get_target_segment(self, frame_range: list[int], frame_shift_aug: float = 0.0) -> torch.Tensor:
        target_data = self.target_provider.get_data(self.actor_name, self.name)
        self.verify_target_data(target_data)
        return self.target_provider.get_segment_data(target_data, self.fps, frame_range, frame_shift_aug)

    def warmup_cache(self):
        _ = self.target_provider.get_data(self.actor_name, self.name)


class MutedShot(Shot):
    def __init__(
        self,
        shot_id: str,
        actor_name: str,
        shot_name: str,
        source_shot_name: str,
        source_frame: int,
        shot_len: int,
        shot_fps: float,
        shot_emotion_name: str,
        shot_start_global: int,
        target_provider: TargetProviderBase,
    ):
        self.id = shot_id
        self.actor_name = actor_name
        self.name = shot_name
        self.source_shot_name = source_shot_name  # source shot from which to take the pose for the muted clip
        self.source_frame = source_frame  # source frame from which to take the pose for the muted clip
        self.len = shot_len
        self.fps = shot_fps
        self.emotion_name = shot_emotion_name
        self.start_global = shot_start_global  # start frame within all shots concatenated
        self.target_provider = target_provider

    def __repr__(self):
        return f"[{self.id}] | len = {self.len} | fps = {self.fps}"

    def get_target_frame_data(self) -> torch.Tensor:
        target_data = self.target_provider.get_data(self.actor_name, self.source_shot_name)
        target_frame_data = self.target_provider.get_muted_frame_data(target_data, self.source_frame, self.emotion_name)
        return target_frame_data

    def get_target_seq(self, frame_idx_seq: list[int], frame_shift_aug: float = 0.0) -> torch.Tensor:
        target_frame_data = self.get_target_frame_data()
        return target_frame_data.unsqueeze(0).repeat(len(frame_idx_seq), 1).float()

    def get_target_segment(self, frame_range: list[int], frame_shift_aug: float = 0.0) -> torch.Tensor:
        target_frame_data = self.get_target_frame_data()
        return self.target_provider.get_muted_segment_data(target_frame_data, self.fps, frame_range)

    def warmup_cache(self):
        _ = self.target_provider.get_data(self.actor_name, self.source_shot_name)


class Clip:
    def __init__(
        self,
        clip_idx: int,
        shot: Shot,
        first_frame: int,
        last_frame: int,
        speaker_name: str,
        audio_lang: str | None,
        audio_fpath_rel: str,
        audio_offset: int,
        audio_provider: AudioProvider,
        phoneme_provider: PhonemeProvider | None,
    ):
        self.clip_idx = clip_idx
        self.shot = shot
        self.first_frame_ori = first_frame
        self.last_frame_ori = last_frame
        self.speaker_name = speaker_name
        self.audio_lang = audio_lang
        self.audio_fpath_rel = audio_fpath_rel
        self.audio_offset = audio_offset  # in frames
        self.audio_provider = audio_provider
        self.phoneme_provider = phoneme_provider
        self.first_frame = first_frame + max(-audio_offset, 0)  # clipping from the front if audio_offset < 0
        self.last_frame = last_frame - max(audio_offset, 0)  # clipping from the end if audio_offset > 0
        self.len = self.last_frame - self.first_frame + 1

    def __repr__(self):
        repr_str = f"[{self.clip_idx}] from shot [{self.shot.id}] | range = [{self.first_frame} - {self.last_frame}]"
        if self.audio_offset != 0:
            repr_str += f" (clipped original range [{self.first_frame_ori} - {self.last_frame_ori}]"
            repr_str += f" due to offset {self.audio_offset:+d})"
        return repr_str

    def get_frame_idx_seq(self, seq_start_in_clip: int, seq_len: int) -> list[int]:
        seq_start_in_shot = self.first_frame + seq_start_in_clip
        return [(seq_start_in_shot + i) for i in range(seq_len)]  # idxs within a shot

    def get_audio_seq(self, t_seq: list[float], t_shift_aug: float = 0.0, rng=None) -> np.ndarray:
        audio_track = self.audio_provider.get_data(self.shot.actor_name, self.audio_fpath_rel)
        t_shift_cfg = self.audio_offset / self.shot.fps
        audio_seq = [self.audio_provider.get_frame_data(audio_track, t, t_shift_aug + t_shift_cfg) for t in t_seq]
        return np.stack(audio_seq).astype(np.float32)

    def get_audio_segment(
        self, t_range: list[float], t_shift_aug: float = 0.0, transform_aug=None, rng=None
    ) -> np.ndarray:
        audio_track = self.audio_provider.get_data(self.shot.actor_name, self.audio_fpath_rel)
        t_shift_cfg = self.audio_offset / self.shot.fps
        return self.audio_provider.get_segment_data(audio_track, t_range, t_shift_aug + t_shift_cfg, transform_aug)

    def get_audio_norm_factor(self) -> np.float32:
        audio_track = self.audio_provider.get_data(self.shot.actor_name, self.audio_fpath_rel)
        return np.float32(audio_track.norm_factor)

    def get_phoneme_seq(self, t_seq: list[float], t_shift_aug: float = 0.0) -> np.ndarray:
        if self.phoneme_provider.phoneme_detector.lang_is_ready(self.audio_lang):
            phonemes = self.phoneme_provider.get_data(self.shot.actor_name, self.audio_fpath_rel, self.audio_lang)
            t_shift_cfg = self.audio_offset / self.shot.fps
            phoneme_seq = [self.phoneme_provider.get_frame_data(phonemes, t, t_shift_aug + t_shift_cfg) for t in t_seq]
            return np.stack(phoneme_seq).astype(np.float32)
        else:
            phoneme_frame_data = self.phoneme_provider.get_dummy_frame_data()
            return np.repeat(phoneme_frame_data[np.newaxis, :], len(t_seq), axis=0).astype(np.float32)

    def get_phoneme_lang(self) -> str:
        if self.phoneme_provider.phoneme_detector.lang_is_ready(self.audio_lang):
            return self.audio_lang
        else:
            return "unsupported"

    def warmup_cache(self):
        self.shot.warmup_cache()
        _ = self.audio_provider.get_data(self.shot.actor_name, self.audio_fpath_rel)
        if self.phoneme_provider is not None:
            if self.phoneme_provider.phoneme_detector.lang_is_ready(self.audio_lang):
                _ = self.phoneme_provider.get_data(self.shot.actor_name, self.audio_fpath_rel, self.audio_lang)


class MutedClip(Clip):
    def __init__(
        self,
        clip_idx: int,
        shot: MutedShot,
        audio_provider: AudioProvider,
        phoneme_provider: PhonemeProvider | None,
    ):
        self.clip_idx = clip_idx
        self.shot = shot
        self.first_frame = 0  # frame idx within a shot
        self.last_frame = shot.len - 1  # frame idx within a shot
        self.audio_provider = audio_provider
        self.phoneme_provider = phoneme_provider
        self.len = shot.len

    def __repr__(self):
        return f"[{self.clip_idx}] from shot [{self.shot.id}] | range = [{self.first_frame} - {self.last_frame}]"

    def get_audio_seq(self, t_seq: list[float], t_shift_aug: float = 0.0, rng=None) -> np.ndarray:
        return self.audio_provider.get_muted_seq_data(t_seq, rng)

    def get_audio_segment(
        self, t_range: list[float], t_shift_aug: float = 0.0, transform_aug=None, rng=None
    ) -> np.ndarray:
        return self.audio_provider.get_muted_segment_data(t_range, rng)

    def get_audio_norm_factor(self) -> np.float32:
        return np.float32(1.0)  # do not normalize muted audio

    def get_phoneme_seq(self, t_seq: list[float], t_shift_aug: float = 0.0) -> np.ndarray:
        phoneme_frame_data = self.phoneme_provider.get_muted_frame_data()
        return np.repeat(phoneme_frame_data[np.newaxis, :], len(t_seq), axis=0).astype(np.float32)

    def get_phoneme_lang(self) -> str:
        return self.phoneme_provider.muted_lang

    def warmup_cache(self):
        self.shot.warmup_cache()


class AnimationDatasetBase(Dataset, ABC):
    def __init__(
        self,
        cfg_train,
        cfg_dataset,
        audio_provider: AudioProvider,
        phoneme_provider: PhonemeProvider | None,
        target_provider: TargetProviderBase,
    ):
        self.cfg_train = cfg_train
        self.cfg_dataset = cfg_dataset
        self.audio_provider = audio_provider
        self.phoneme_provider = phoneme_provider
        self.target_provider = target_provider
        self.rng = np.random.default_rng(self.cfg_train.RNG_SEED_DATASET)
        self.shots: OrderedDict[str, Shot] = OrderedDict()
        self.clips: list[Clip] = []
        self.len: int = 0
        logging.info(f"Using emotions: {self.cfg_dataset.SHOT_EMOTION_NAMES}")
        self.add_shots()
        self.add_clips()
        self.add_muted_shots_and_clips()

    def add_shots(self):
        shot_start_global = 0
        for actor_name, shot_name, shot_len, shot_fps, shot_emotion_name in self.target_provider.shot_list:
            shot_id = gen_id(actor_name, shot_name, self.cfg_dataset.ACTOR_NAMES)
            utils.validate_actor_name(actor_name, self.cfg_dataset.ACTOR_NAMES, msg_prefix=shot_id)
            utils.validate_shot_emotion_name(shot_emotion_name, self.cfg_dataset.SHOT_EMOTION_NAMES, msg_prefix=shot_id)
            shot = Shot(
                shot_id=shot_id,
                actor_name=actor_name,
                shot_name=shot_name,
                shot_len=shot_len,
                shot_fps=shot_fps,
                shot_emotion_name=shot_emotion_name,
                shot_start_global=shot_start_global,
                target_provider=self.target_provider,
            )
            if shot.id in self.shots.keys():
                logging.warning(f"{shot.id} : Shot appears in the list more than once")
            self.shots[shot.id] = shot
            logging.info(f"Added Shot: {shot}")
            shot_start_global += shot_len

    def add_clips(self):
        if len(self.cfg_train.CLIPS) == 0:
            raise ValueError("CLIPS list is empty")

        for clip_idx, clip_info in enumerate(self.cfg_train.CLIPS):
            actor_name, shot_name, shot_range, speaker_name, audio_lang, audio_fpath_rel, audio_offset = clip_info
            first_frame, last_frame = shot_range
            shot_id = gen_id(actor_name, shot_name, self.cfg_dataset.ACTOR_NAMES)
            if shot_id not in self.shots.keys():
                raise ValueError(f'Shot "{shot_id}" from CLIPS is not presented in the shot list')
            shot = self.shots[shot_id]
            if last_frame > shot.len - 1:
                logging.warning(f"{shot.id} : Clip last_frame {last_frame} is out of shot range [0 - {shot.len-1}]")
            self.clips.append(
                Clip(
                    clip_idx=clip_idx,
                    shot=shot,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    speaker_name=speaker_name,
                    audio_lang=audio_lang,
                    audio_fpath_rel=audio_fpath_rel,
                    audio_offset=audio_offset,
                    audio_provider=self.audio_provider,
                    phoneme_provider=self.phoneme_provider,
                )
            )
            logging.info(f"Added Clip: {self.clips[-1]}")

    def add_muted_shots_and_clips(self):
        last_shot = self.shots[list(self.shots.keys())[-1]]
        shot_start_global = last_shot.start_global + last_shot.len
        clip_idx = self.clips[-1].clip_idx + 1
        for actor_name, source, muted_shot_len, shot_emotion_name in self.cfg_train.AUG_MUTED_SHOTS:
            if not (isinstance(source, (tuple, list)) and len(source) == 2):
                raise ValueError(f"Wrong source in AUG_MUTED_SHOTS: {source}. Should be (source_shot, source_frame)")
            source_shot_name, source_frame = source
            shot_name = f"muted_{source_shot_name}_{source_frame}"
            shot_id = gen_id(actor_name, shot_name, self.cfg_dataset.ACTOR_NAMES)
            utils.validate_actor_name(actor_name, self.cfg_dataset.ACTOR_NAMES, msg_prefix=shot_id)
            utils.validate_shot_emotion_name(shot_emotion_name, self.cfg_dataset.SHOT_EMOTION_NAMES, msg_prefix=shot_id)
            shot_fps = self.cfg_dataset.CACHE_FPS[actor_name]
            shot = MutedShot(
                shot_id=shot_id,
                actor_name=actor_name,
                shot_name=shot_name,
                source_shot_name=source_shot_name,
                source_frame=source_frame,
                shot_len=muted_shot_len,
                shot_fps=shot_fps,
                shot_emotion_name=shot_emotion_name,
                shot_start_global=shot_start_global,
                target_provider=self.target_provider,
            )
            if shot.id in self.shots.keys():
                logging.warning(f"{shot.id} : Muted Shot appears in the list more than once")
            self.shots[shot.id] = shot
            logging.info(f"Added Muted Shot: {shot}")
            self.clips.append(
                MutedClip(
                    clip_idx=clip_idx,
                    shot=shot,
                    audio_provider=self.audio_provider,
                    phoneme_provider=self.phoneme_provider,
                )
            )
            logging.info(f"Added Muted Clip: {self.clips[-1]}")
            shot_start_global += shot.len
            clip_idx += 1

    def emotion_name2vec(self, shot_emotion_name: str) -> np.ndarray:
        utils.validate_shot_emotion_name(shot_emotion_name, self.cfg_dataset.SHOT_EMOTION_NAMES)
        network_emotion_names = utils.get_network_emotion_names(
            self.cfg_dataset.SHOT_EMOTION_NAMES, self.cfg_train.SHOT_EMOTION_NAME_FOR_ALL_ZEROS
        )
        emotion_vec = np.zeros((len(network_emotion_names)), dtype=np.float32)
        if (self.cfg_train.SHOT_EMOTION_NAME_FOR_ALL_ZEROS is None) or (
            shot_emotion_name != self.cfg_train.SHOT_EMOTION_NAME_FOR_ALL_ZEROS
        ):
            emotion_idx = network_emotion_names.index(shot_emotion_name)
            emotion_vec[emotion_idx] = 1.0
        return emotion_vec

    def actor_name2vec(self, actor_name: str) -> np.ndarray:
        utils.validate_actor_name(actor_name, self.cfg_dataset.ACTOR_NAMES)
        actor_vec = np.zeros((len(self.cfg_dataset.ACTOR_NAMES)), dtype=np.float32)
        actor_idx = self.cfg_dataset.ACTOR_NAMES.index(actor_name)
        actor_vec[actor_idx] = 1.0
        return actor_vec

    @abstractmethod
    def preload(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> utils.EasyDict:
        pass


class AnimationSequenceDataset(AnimationDatasetBase):
    def __init__(
        self,
        cfg_train,
        cfg_dataset,
        audio_provider: AudioProvider,
        phoneme_provider: PhonemeProvider | None,
        target_provider: TargetProviderBase,
    ):
        super().__init__(cfg_train, cfg_dataset, audio_provider, phoneme_provider, target_provider)
        self.num_seqs_per_clip_cum: list[int] = []
        self.total_shot_frames: int = 0
        self.calc_ds_info()

    def calc_ds_info(self):
        num_seqs_per_clip_all = []
        for clip in self.clips:
            num_seqs_per_clip = (clip.len - self.cfg_train.SEQ_LEN) // self.cfg_train.SEQ_STRIDE + 1
            num_seqs_per_clip_all.append(num_seqs_per_clip)
        self.len = np.sum(num_seqs_per_clip_all)  # Pytorch dataset length (total number of sequences)
        self.num_seqs_per_clip_cum = list(np.cumsum(num_seqs_per_clip_all))
        self.total_shot_frames = np.sum([shot.len for shot in self.shots.values()])

    def preload(self):
        logging.info("Preloading dataset into cache...")
        for clip in self.clips:
            clip.warmup_cache()
        logging.info(f"Loaded shots: {len(self.target_provider.cache)}")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, global_seq_idx: int) -> utils.EasyDict[np.ndarray | torch.Tensor | np.float32 | str]:
        clip_idx = np.searchsorted(self.num_seqs_per_clip_cum, global_seq_idx + 1)
        seq_idx_in_clip = global_seq_idx - self.num_seqs_per_clip_cum[clip_idx - 1] if clip_idx > 0 else global_seq_idx
        seq_start_in_clip = seq_idx_in_clip * self.cfg_train.SEQ_STRIDE
        clip = self.clips[clip_idx]

        # Shift Augmentation
        frame_shift_aug = (self.rng.uniform() - 0.5) * self.cfg_train.AUGMENT["timeshift"]
        t_shift_aug = frame_shift_aug / clip.shot.fps

        frame_idx_seq = clip.get_frame_idx_seq(seq_start_in_clip, self.cfg_train.SEQ_LEN)  # idxs within a shot
        t_seq = clip.shot.get_t_seq(frame_idx_seq)
        audio_seq = clip.get_audio_seq(t_seq, t_shift_aug, self.rng)
        target_seq = clip.shot.get_target_seq(frame_idx_seq, frame_shift_aug)
        global_frame_idx_seq = clip.shot.get_global_frame_idx_seq(frame_idx_seq)
        explicit_emo = self.emotion_name2vec(clip.shot.emotion_name)
        explicit_emo_seq = np.stack([explicit_emo] * self.cfg_train.SEQ_LEN).astype(np.float32)
        audio_norm_factor = clip.get_audio_norm_factor()

        item = {
            "x": audio_seq,
            "y": target_seq,
            "global_frame_idx": global_frame_idx_seq,
            "explicit_emo": explicit_emo_seq,
            "explicit_emo_name": clip.shot.emotion_name,
            "x_norm_factor": audio_norm_factor,
        }

        if self.phoneme_provider is not None:
            phoneme_seq = clip.get_phoneme_seq(t_seq, t_shift_aug)
            phoneme_lang = clip.get_phoneme_lang()
            item.update(
                {
                    "phonemes": phoneme_seq,
                    "lang": phoneme_lang,
                }
            )

        return utils.EasyDict(item)


class AnimationSegmentDataset(AnimationDatasetBase):
    def __init__(
        self,
        cfg_train,
        cfg_dataset,
        audio_provider: AudioProvider,
        phoneme_provider: PhonemeProvider | None,
        target_provider: TargetProviderBase,
    ):
        super().__init__(cfg_train, cfg_dataset, audio_provider, phoneme_provider, target_provider)
        self.audio_transform_aug: audiomentations.Compose | None = None
        self.calc_ds_info()
        self.compose_aug()

    def calc_ds_info(self):
        self.len = len(self.clips)  # Pytorch dataset length (total number of clips)

    def preload(self):
        pass

    def compose_aug(self):
        transforms = []
        if "pitch_shift" in self.cfg_train.AUGMENT.keys():
            transforms.append(audiomentations.PitchShift(**self.cfg_train.AUGMENT["pitch_shift"]))
        if len(transforms) > 0:
            self.audio_transform_aug = audiomentations.Compose(transforms)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, clip_idx: int) -> utils.EasyDict[np.ndarray | torch.Tensor | np.float32 | str]:
        clip = self.clips[clip_idx]

        # Shift Augmentation
        frame_shift_aug = (self.rng.uniform() - 0.5) * self.cfg_train.AUGMENT["timeshift"]
        t_shift_aug = frame_shift_aug / clip.shot.fps

        frame_range = [clip.first_frame, clip.last_frame]
        t_range = clip.shot.frame_range_to_time_range(frame_range)
        audio_segment = clip.get_audio_segment(t_range, t_shift_aug, self.audio_transform_aug, self.rng)
        target_segment = clip.shot.get_target_segment(frame_range, frame_shift_aug)
        emotion_vec = self.emotion_name2vec(clip.shot.emotion_name)
        actor_vec = self.actor_name2vec(clip.shot.actor_name)

        item = {
            "x": audio_segment,
            "y": target_segment,
            "emotion_vec": emotion_vec,
            "emotion_name": clip.shot.emotion_name,
            "actor_vec": actor_vec,
            "actor_name": clip.shot.actor_name,
        }
        return utils.EasyDict(item)


def extract_random_subsegment(
    vertices: torch.Tensor,
    audio: torch.Tensor,
    sample_rate: int = 16000,
    fps: float = 30.0,
    min_frame: int = 30,
    max_frame: int = 150,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts random segments from vertices, and audio tensors.
    """
    num_frames = np.random.randint(min_frame, max_frame + 1)
    num_frames = min(num_frames, vertices.shape[1])
    duration_seconds = num_frames / fps

    num_samples = int(duration_seconds * sample_rate)
    max_start_frame = vertices.shape[1] - num_frames
    start_frame = np.random.randint(0, max_start_frame + 1)
    start_sample = max(int(start_frame * (sample_rate / fps)), 0)

    audio_segment = audio[:, start_sample : start_sample + num_samples]
    vertices_segment = vertices[:, start_frame : start_frame + num_frames, :]

    return vertices_segment, audio_segment
