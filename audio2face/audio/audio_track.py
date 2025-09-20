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
import math
import numpy as np
import scipy.signal


class AudioTrack:
    def __init__(self, data: np.ndarray | None = None, samplerate: int = 48000) -> None:
        self.data = data.astype(np.float32) if data is not None else np.zeros(0, dtype=np.float32)
        self.samplerate = samplerate
        self.norm_factor = 1.0
        assert self.data.ndim == 1
        assert self.samplerate > 0

    def get_length(self) -> float:
        return float(self.data.size) / float(self.samplerate)

    def get_num_samples(self) -> int:
        return self.data.size

    def sec_to_sample(self, sec: float) -> int:
        return int(round(sec * self.samplerate))

    def sample_to_sec(self, sample: int) -> float:
        return float(sample) / float(self.samplerate)

    def get_padded_buffer(self, ofs: int, length: int) -> np.ndarray:
        if ofs >= 0 and ofs + length <= self.data.size:
            return self.data[ofs : ofs + length]
        res = np.zeros(length, dtype=self.data.dtype)
        begin = max(0, -ofs)
        end = min(length, self.data.size - ofs)
        if begin < end:
            res[begin:end] = self.data[ofs + begin : ofs + end]
        return res

    def get_resampled_padded_buffer(
        self, input_buffer_pos: int, resampled_ofs: int, resampled_len: int, new_samplerate: int
    ) -> np.ndarray:
        if self.samplerate == new_samplerate:
            ofs = input_buffer_pos - resampled_ofs
            return self.get_padded_buffer(ofs, resampled_len)
        resample_ratio = float(new_samplerate) / self.samplerate
        resample_up = max(int(round(min(resample_ratio, 1) * 1000)), 1)
        resample_down = max(int(round(resample_up / resample_ratio)), 1)
        input_buffer_len = int(math.ceil(float(resampled_len) * resample_down / resample_up))
        input_buffer_ofs = int(round(float(resampled_ofs) * resample_down / resample_up))
        ofs = input_buffer_pos - input_buffer_ofs
        buffer_track = AudioTrack(self.get_padded_buffer(ofs, input_buffer_len), self.samplerate)
        buffer_track.resample(new_samplerate)
        return buffer_track.get_padded_buffer(0, resampled_len)

    def pad(self, pad_sec: float) -> None:
        padding = int(pad_sec * self.samplerate)
        self.data = np.concatenate((np.zeros((padding), np.float32), self.data))

    def resample(self, new_samplerate: int) -> None:
        if self.samplerate == new_samplerate:
            return
        resample_ratio = float(new_samplerate) / self.samplerate
        resample_up = max(int(round(min(resample_ratio, 1) * 1000)), 1)
        resample_down = max(int(round(resample_up / resample_ratio)), 1)
        self.data = scipy.signal.resample_poly(self.data.astype(np.float32), resample_up, resample_down).astype(
            np.float32
        )
        self.samplerate = new_samplerate

    def normalize(self, threshold: float = 0.01) -> None:
        maxabs = np.max(np.abs(self.data))
        if maxabs > threshold:
            self.data /= max(maxabs, 1.0e-8)

    def update_norm_factor(self, threshold: float = 0.01) -> None:
        maxabs = np.max(np.abs(self.data))
        if maxabs > threshold:
            self.norm_factor = max(maxabs, 1.0e-8)
        else:
            self.norm_factor = 1.0  # We should not normalize if maxabs is "small"


def read_data(data: np.ndarray, samplerate: int, pad: int = 0) -> AudioTrack:
    data = data.astype(np.float32)
    # Convert to mono.
    if len(data.shape) > 1:
        assert len(data.shape) == 2
        data = np.average(data, axis=1)
    # Normalize volume.
    data /= max(np.max(abs(data)), 1.0e-8)
    if pad > 0:
        data = np.concatenate((np.zeros((pad), np.float32), data))
    return AudioTrack(data, samplerate)
