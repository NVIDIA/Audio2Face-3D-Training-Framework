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
import soundfile

import torch
import torchaudio

from audio2face.audio import AudioTrack


def read_audio_track(fpath: str) -> AudioTrack:
    data, samplerate = soundfile.read(fpath, dtype="float32")
    if len(data.shape) > 1:
        data = np.average(data, axis=1)  # convert to mono
    return AudioTrack(data, samplerate)


def read_and_preproc_audio_track(
    fpath: str,
    preproc_method: str = "nva2f",
    new_samplerate: int | None = None,
) -> AudioTrack:
    if preproc_method == "nva2f":
        track = read_audio_track(fpath)
        track.normalize(threshold=0.01)
        if new_samplerate is not None:
            track.resample(new_samplerate)
    elif preproc_method == "w2v":
        track = read_audio_track(fpath)
        data, samplerate = torch.from_numpy(track.data), track.samplerate
        if new_samplerate is not None:
            data = torchaudio.functional.resample(data, samplerate, new_samplerate)
            samplerate = new_samplerate
        track = AudioTrack(data.numpy(), samplerate)
        track.update_norm_factor(threshold=0.01)
    else:
        raise ValueError(f"Unsupported audio preprocessing method: {preproc_method}")
    return track


def generate_audio_noise(
    buffer_len: int,
    samplerate: int,
    noise_type: str | None = None,
    noise_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if noise_type is None:
        return np.zeros(buffer_len, dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng()
    if noise_type == "gauss":
        return rng.normal(0.0, 1.0, buffer_len) * noise_scale
    elif noise_type == "mic":
        # Simulate a microphone noise by creating a spectrogram with given distribution of frequencies
        nperseg = 256
        size_f = nperseg // 2 + 1
        size_t = math.ceil((buffer_len - nperseg) / (nperseg - nperseg // 2)) + 3
        frequencies = np.linspace(0, samplerate // 2, size_f)
        Sxx_log_mean = 39.0 * np.exp(-frequencies / 300.0) - 121.0
        Sxx_log_std = 5.85
        Sxx = []
        for _ in range(size_t):
            Sxx_log_i = rng.normal(loc=Sxx_log_mean, scale=Sxx_log_std)
            Sxx_i = np.sqrt(np.power(10.0, (Sxx_log_i / 10.0)))
            Sxx.append(Sxx_i)
        Sxx = np.stack(Sxx).T
        phase = rng.random(size=(size_f, size_t)) * 2 * np.pi
        Zxx = Sxx * np.exp(1j * phase)
        _, audio_data = scipy.signal.istft(Zxx, samplerate)
        return audio_data[:buffer_len] * noise_scale
    else:
        raise ValueError(f"Unsupported audio noise type: {noise_type}")
