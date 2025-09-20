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
import pickle
import numpy as np

from audio2face import utils
from audio2face.dataset import Clip


class ImplicitEmotionManager:
    def __init__(self, emo_db: np.ndarray | None = None, emo_specs: dict | None = None) -> None:
        self.emo_db = emo_db
        self.emo_specs = emo_specs
        self.compactified = False

    def load(self, fpath: str) -> None:
        with open(fpath, "rb") as f:
            emo_data = pickle.load(f)
        self.emo_db = emo_data["emo_db"]
        self.emo_specs = emo_data["emo_specs"]
        self.compactified = emo_data["compactified"]

    def save_pkl(self, fpath: str) -> None:
        emo_data = {
            "emo_db": self.emo_db,
            "emo_specs": self.emo_specs,
            "compactified": self.compactified,
        }
        with open(fpath, "wb") as f:
            pickle.dump(emo_data, f)

    def load_bin(self, fpath: str) -> None:
        with open(fpath, "rb") as f:
            data = f.read()
        idx = 0
        num_shots = utils.bytes2int(data[idx : idx + 4])
        idx += 4
        self.emo_specs = {}
        for _ in range(num_shots):
            shot_name_len = utils.bytes2int(data[idx : idx + 4])
            idx += 4
            shot_name = data[idx : idx + shot_name_len].decode()
            idx += shot_name_len
            first_frame = utils.bytes2int(data[idx : idx + 4])
            idx += 4
            num_frames = utils.bytes2int(data[idx : idx + 4])
            idx += 4
            self.emo_specs[shot_name] = (first_frame, num_frames)
        emo_len = utils.bytes2int(data[idx : idx + 4])
        idx += 4
        emo_db_size = utils.bytes2int(data[idx : idx + 4])
        idx += 4
        self.emo_db = np.frombuffer(data[idx : idx + emo_db_size * 4], dtype=np.float32).reshape(-1, emo_len)
        self.compactified = None  # unknown

    def save_bin(self, fpath: str) -> None:
        emo_shots = list(self.emo_specs.items())
        data = b""
        data += utils.int2bytes(len(emo_shots))
        for emo_shot in emo_shots:
            data += utils.int2bytes(len(emo_shot[0]))
            data += emo_shot[0].encode()
            data += utils.int2bytes(emo_shot[1][0])
            data += utils.int2bytes(emo_shot[1][1])
        data += utils.int2bytes(self.emo_db.shape[1])
        data += utils.int2bytes(self.emo_db.size)
        data += self.emo_db.tobytes()
        with open(fpath, "wb") as f:
            f.write(data)

    def load_npz(self, fpath: str) -> None:
        npz_data = np.load(fpath)
        self.emo_db = npz_data["emo_db"]
        self.compactified = None  # unknown
        self.emo_specs = {}

        emo_spec_names = [name.decode("utf-8") for name in npz_data["emo_spec_names"]]
        for name, start, size in zip(emo_spec_names, npz_data["emo_spec_start"], npz_data["emo_spec_size"]):
            self.emo_specs[name] = (start, size)

    def save_npz(self, fpath: str) -> None:
        emo_data = {}
        emo_data["emo_db"] = self.emo_db
        emo_data["emo_spec_names"] = []
        emo_data["emo_spec_start"] = []
        emo_data["emo_spec_size"] = []

        emo_specs = self.emo_specs
        emo_specs_sorted = sorted([k for k in emo_specs], key=lambda x: emo_specs[x][0])
        for k in emo_specs_sorted:
            start, size = emo_specs[k]
            emo_data["emo_spec_names"].append(k)
            emo_data["emo_spec_start"].append(start)
            emo_data["emo_spec_size"].append(size)

        emo_data["emo_spec_names"] = np.array(emo_data["emo_spec_names"], dtype="S")
        emo_data["emo_spec_start"] = np.array(emo_data["emo_spec_start"], dtype=np.int32)
        emo_data["emo_spec_size"] = np.array(emo_data["emo_spec_size"], dtype=np.int32)
        np.savez(fpath, **emo_data)

    def emo_spec_to_idx(self, emo_spec: tuple[str, int]) -> int:
        if self.emo_specs is None:
            raise RuntimeError("ImplicitEmotionManager is not initialized")
        emo_shot, emo_frame = emo_spec
        first_frame, num_frames = self.emo_specs[emo_shot]
        if emo_frame < 0 or emo_frame >= num_frames:
            raise RuntimeError(
                "Emotion Frame {} is out of range [{}, {}] for Shot {}".format(emo_frame, 0, num_frames - 1, emo_shot)
            )
        idx = first_frame + emo_frame
        return idx

    def get_emotion_vector(self, emo_spec: tuple[str, int]) -> np.ndarray:
        if self.emo_db is None:
            raise RuntimeError("ImplicitEmotionManager is not initialized")
        global_idx = self.emo_spec_to_idx(emo_spec)
        return self.emo_db[global_idx, :]

    def get_shot_matrix(self, shot_id: str) -> np.ndarray:
        start, size = self.emo_specs[shot_id]
        return self.emo_db[start : start + size]

    def compactify(self, dataset_clips: list[Clip]) -> None:
        if self.compactified:
            return
        compact_emo_db = np.zeros((0, self.emo_db.shape[1]), dtype=self.emo_db.dtype)
        compact_emo_specs = {}
        sorted_shots = sorted(self.emo_specs.items(), key=lambda spec: spec[1][0])
        compact_shot_start_global = 0
        for shot_id, _ in sorted_shots:
            shot_ranges = []
            for clip in dataset_clips:
                if clip.shot.id == shot_id:
                    shot_ranges.append((clip.first_frame, clip.last_frame))
            if len(shot_ranges) > 0:  # if this shot is covered by any clips
                shot_ranges = utils.merge_ranges(shot_ranges)
                shot_matrix = self.get_shot_matrix(shot_id)
                compact_shot_matrix = utils.get_merged_submatrix_from_ranges(shot_matrix, shot_ranges)
                compact_emo_db = np.concatenate((compact_emo_db, compact_shot_matrix), axis=0)
                compact_shot_len = len(compact_shot_matrix)
                compact_emo_specs[shot_id] = (compact_shot_start_global, compact_shot_len)
                compact_shot_start_global += compact_shot_len
        self.emo_db = compact_emo_db
        self.emo_specs = compact_emo_specs
        self.compactified = True
