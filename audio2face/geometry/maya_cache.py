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
import array
from typing import List, Union
import xml.etree.ElementTree as ET
import numpy as np


class MayaCacheReader:
    def __init__(self, xml_path: str) -> None:
        self.xml_path = xml_path
        self.base_name = os.path.splitext(os.path.basename(xml_path))[0]
        self.directory = os.path.dirname(xml_path)
        self.mc_path = os.path.join(self.directory, f"{self.base_name}.mc")
        self.cache_type = ""
        self.start_time = 0
        self.end_time = 0
        self.time_per_frame = 0
        self.version = ""
        self.channels = []
        self._parse_xml()

    def _parse_xml(self) -> None:
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        cache_type_elem = root.find("cacheType")
        if cache_type_elem is not None:
            self.cache_type = cache_type_elem.get("Type", "")

        time_elem = root.find("time")
        if time_elem is not None:
            time_range = time_elem.get("Range", "0-0")
            start_str, end_str = time_range.split("-")
            self.start_time = int(start_str)
            self.end_time = int(end_str)

        time_per_frame_elem = root.find("cacheTimePerFrame")
        if time_per_frame_elem is not None:
            self.time_per_frame = int(time_per_frame_elem.get("TimePerFrame", "0"))

        version_elem = root.find("cacheVersion")
        if version_elem is not None:
            self.version = version_elem.get("Version", "")

        channels_elem = root.find("Channels")
        if channels_elem is not None:
            for channel_elem in channels_elem:
                if channel_elem.tag.startswith("channel"):
                    channel_info = {
                        "name": channel_elem.get("ChannelName", ""),
                        "type": channel_elem.get("ChannelType", ""),
                        "interpretation": channel_elem.get("ChannelInterpretation", ""),
                        "sampling_type": channel_elem.get("SamplingType", ""),
                        "sampling_rate": int(channel_elem.get("SamplingRate", "0")),
                        "start_time": int(channel_elem.get("StartTime", "0")),
                        "end_time": int(channel_elem.get("EndTime", "0")),
                    }
                    self.channels.append(channel_info)

    def _need_byte_swap(self) -> bool:
        return sys.platform.startswith(("win", "linux"))

    def _read_int32(self, file_handle) -> int:
        int_array = array.array("i" if array.array("l").itemsize > 4 else "l")
        int_array.fromfile(file_handle, 1)
        if self._need_byte_swap():
            int_array.byteswap()
        return int_array[0]

    def read_frames(self):
        if not os.path.exists(self.mc_path):
            raise FileNotFoundError(f"Binary cache file not found: {self.mc_path}")

        with open(self.mc_path, "rb") as f:
            header_tag = f.read(4)
            if header_tag != b"FOR4":
                raise ValueError(f"Invalid cache file format: expected FOR4, got {header_tag}")

            header_size = self._read_int32(f)
            f.read(header_size)

            while True:
                try:
                    block_tag = f.read(4)
                    if len(block_tag) != 4:
                        break

                    if block_tag != b"FOR4":
                        raise ValueError(f"Invalid block format: expected FOR4, got {block_tag}")

                    block_size = self._read_int32(f)

                    mych_tag = f.read(4)
                    if mych_tag != b"MYCH":
                        raise ValueError(f"Invalid block header: expected MYCH, got {mych_tag}")

                    time_tag = f.read(4)
                    if time_tag != b"TIME":
                        raise ValueError(f"Invalid time tag: expected TIME, got {time_tag}")

                    self._read_int32(f)  # time_size
                    self._read_int32(f)  # frame_time

                    bytes_read = 16

                    while bytes_read < block_size:
                        chnm_tag = f.read(4)
                        if chnm_tag != b"CHNM":
                            raise ValueError(f"Invalid channel name tag: expected CHNM, got {chnm_tag}")

                        name_size = self._read_int32(f)
                        padded_size = (name_size + 3) & ~3
                        f.read(name_size)
                        if padded_size > name_size:
                            f.read(padded_size - name_size)

                        bytes_read += 8 + padded_size

                        size_tag = f.read(4)
                        if size_tag != b"SIZE":
                            raise ValueError(f"Invalid size tag: expected SIZE, got {size_tag}")

                        self._read_int32(f)  # size_field_size
                        array_length = self._read_int32(f)
                        bytes_read += 12

                        data_format = f.read(4)
                        buffer_length = self._read_int32(f)
                        bytes_read += 8

                        if data_format == b"FVCA":
                            expected_size = array_length * 3 * 4
                            if buffer_length != expected_size:
                                raise ValueError(f"Buffer size mismatch: expected {expected_size}, got {buffer_length}")

                            float_data = array.array("f")
                            float_data.fromfile(f, array_length * 3)
                            if self._need_byte_swap():
                                float_data.byteswap()

                            vertex_array = np.frombuffer(float_data, dtype=np.float32).reshape(-1, 3)
                            bytes_read += buffer_length
                            yield vertex_array

                        elif data_format == b"DVCA":
                            expected_size = array_length * 3 * 8
                            if buffer_length != expected_size:
                                raise ValueError(f"Buffer size mismatch: expected {expected_size}, got {buffer_length}")

                            double_data = array.array("d")
                            double_data.fromfile(f, array_length * 3)
                            if self._need_byte_swap():
                                double_data.byteswap()

                            vertex_array = np.frombuffer(double_data, dtype=np.float64).reshape(-1, 3)
                            bytes_read += buffer_length
                            yield vertex_array

                        else:
                            raise ValueError(f"Unsupported data format: {data_format}")

                except EOFError:
                    break


class MayaCacheWriter:
    def __init__(self, base_filename: str, fps: float, dtype: np.dtype = np.float64) -> None:
        self.base_filename = os.path.splitext(base_filename)[0]
        self.fps = fps
        self.dtype = dtype
        self.frame_time = 6000.0 / fps

        if dtype == np.float64:
            self.array_type = "d"
            self.vertex_size = 24
            self.channel_type = "DoubleVectorArray"
            self.data_tag = b"DVCA"
        elif dtype == np.float32:
            self.array_type = "f"
            self.vertex_size = 12
            self.channel_type = "FloatVectorArray"
            self.data_tag = b"FVCA"
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

    def _write_int32(self, file_handle, value: int) -> None:
        int_array = array.array("i" if array.array("l").itemsize > 4 else "l")
        int_array.insert(0, value)
        int_array.byteswap()
        int_array.tofile(file_handle)

    def write_xml(self, frame_count: int, extra_info: List[str] | None = None) -> None:
        if extra_info is None:
            extra_info = ["", "", ""]

        end_time = int(round((frame_count - 1) * self.frame_time))
        time_per_frame = int(round(self.frame_time))

        xml_content = f"""<?xml version="1.0"?>
<Autodesk_Cache_File>
  <cacheType Type="OneFile" Format="mcc"/>
  <time Range="0-{end_time}"/>
  <cacheTimePerFrame TimePerFrame="{time_per_frame}"/>
  <cacheVersion Version="2.0"/>
  <extra>maya 2018 x64</extra>
  <extra>username</extra>
  <regression>{extra_info[0]}</regression>
  <regression>{extra_info[1]}</regression>
  <regression>{extra_info[2]}</regression>
  <Channels>
         <channel0 ChannelName="head_meshShape" ChannelType="{self.channel_type}"
               ChannelInterpretation="positions" SamplingType="Regular"
               SamplingRate="{time_per_frame}" StartTime="0" EndTime="{end_time}"/>
  </Channels>
</Autodesk_Cache_File>"""

        with open(f"{self.base_filename}.xml", "w") as f:
            f.write(xml_content)

    def write_binary(self, frames: List[Union[array.array, np.ndarray]]) -> None:
        with open(f"{self.base_filename}.mc", "wb") as f:
            f.write(b"FOR4")
            self._write_int32(f, 40)
            f.write(
                b"CACHVRSN\x00\x00\x00\x040.1\x00STIM\x00\x00\x00\x04\x00\x00\x00\x00"
                b"ETIM\x00\x00\x00\x04\x00\x00\x00\x01"
            )

            for frame_idx, frame_data in enumerate(frames):
                if isinstance(frame_data, np.ndarray):
                    vertex_count = len(frame_data) // 3
                    data_array = array.array(self.array_type, frame_data.flatten())
                else:
                    vertex_count = len(frame_data) // 3
                    data_array = frame_data

                f.write(b"FOR4")
                block_size = 60 + vertex_count * self.vertex_size
                self._write_int32(f, block_size)

                f.write(b"MYCH")
                f.write(b"TIME")
                self._write_int32(f, 4)
                self._write_int32(f, int(round(self.frame_time * frame_idx)))

                f.write(b"CHNM")
                self._write_int32(f, 15)
                f.write(b"head_meshShape\x00\x00")

                f.write(b"SIZE")
                self._write_int32(f, 4)
                self._write_int32(f, vertex_count)

                f.write(self.data_tag)
                self._write_int32(f, vertex_count * self.vertex_size)

                data_array.byteswap()
                data_array.tofile(f)


def read_cache_mc(fpath: str) -> np.ndarray:
    reader = MayaCacheReader(fpath)

    if reader.cache_type != "OneFile":
        raise ValueError(f"Unsupported cache type: {reader.cache_type}")

    frames = list(reader.read_frames())
    if not frames:
        raise ValueError("No frame data found in cache file")

    return np.stack([np.array(frame, np.float32).reshape(-1, 3) for frame in frames], axis=0)


def write_cache_mc(
    filename: str, frames: List[array.array], fps: float, dtype: np.dtype = np.float64, extra: List[str] | None = None
) -> None:
    if extra is None:
        extra = ["", "", ""]

    writer = MayaCacheWriter(filename, fps, dtype)
    writer.write_xml(len(frames), extra)
    writer.write_binary(frames)


def export_animation(animation: List[np.ndarray], fps: float, output_fpath: str) -> None:
    base_dir = os.path.dirname(output_fpath)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir)

    dtype = animation[0].dtype
    if dtype == np.float64:
        typecode = "d"
    elif dtype == np.float32:
        typecode = "f"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    cache_frames = []
    for frame in animation:
        frame_array = array.array(typecode, frame.flatten().tolist())
        cache_frames.append(frame_array)

    write_cache_mc(output_fpath, cache_frames, fps, dtype=dtype)
