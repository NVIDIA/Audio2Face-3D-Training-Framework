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

import torch
import torch.onnx

from audio2face import utils
from audio2face.networks.base import NetworkBaseDiffusion
from audio2face.infer_diffusion import InferenceEngine


class Converter(InferenceEngine):
    def __init__(
        self,
        training_run_name_full: str,
        cfg_train_mod: dict | None = None,
        cfg_dataset_mod: dict | None = None,
        cfg_inference_mod: dict | None = None,
    ) -> None:
        self.training_run_name_full = training_run_name_full
        self.cfg_train_mod = cfg_train_mod
        self.cfg_dataset_mod = cfg_dataset_mod
        self.cfg_inference_mod = cfg_inference_mod

    def setup(self) -> None:
        super().setup()
        self.deploy_dir = os.path.normpath(os.path.join(self.training_artifact_dir, "deploy"))

        self.wrap_model = WrapModel(
            self.cfg_train, self.cfg_dataset, self.cfg_inference, self.network, self.diffusion, self.sample_fn
        )
        self.num_frames = (
            self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["block_frame_size"]
            + self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["left_truncate"]
            + self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["right_truncate"]
        )
        self.total_dim = (
            self.wrap_model.model.skin_dim
            + self.wrap_model.model.tongue_dim
            + self.wrap_model.model.jaw_dim
            + self.wrap_model.model.eye_dim
        )

    def convert_model(self) -> None:
        wrapped_model = self.wrap_model
        num_diffusion_steps = len(wrapped_model.diffusion.use_timesteps)
        batch_size = 3
        num_actors = len(self.cfg_dataset.ACTOR_NAMES)
        window = torch.randn(
            batch_size,
            self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["window_size"],
            dtype=torch.float32,
        ).to(torch.device("cuda"))
        actor_vec = torch.randn(batch_size, num_actors, dtype=torch.float32).to(torch.device("cuda"))
        emo_len = len(
            utils.get_network_emotion_names(
                self.cfg_dataset.SHOT_EMOTION_NAMES, self.cfg_train.SHOT_EMOTION_NAME_FOR_ALL_ZEROS
            )
        )
        if self.cfg_inference.USE_PER_FRAME_EMO_LABEL:
            emotion_vec = torch.randn(
                batch_size,
                self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["block_frame_size"],
                emo_len,
                dtype=torch.float32,
            ).to(torch.device("cuda"))
        else:
            emotion_vec = torch.randn(batch_size, emo_len, dtype=torch.float32).to(torch.device("cuda"))
        h_gru_all = torch.randn(
            num_diffusion_steps,
            self.cfg_train["NETWORK_HYPER_PARAMS"]["num_gru_layers"],
            batch_size,
            self.cfg_train["NETWORK_HYPER_PARAMS"]["gru_feature_dim"],
            dtype=torch.float32,
        ).to(torch.device("cuda"))

        if self.cfg_inference.USE_EXTERNAL_NOISE_INPUT:
            noise = torch.randn([batch_size, num_diffusion_steps + 1, self.num_frames, self.total_dim], device="cuda")
        else:
            noise = None  # use model with internal generated random noise

        onnx_model_fpath = os.path.join(self.deploy_dir, "network.onnx")
        torch.onnx.export(
            wrapped_model,
            (window, actor_vec, emotion_vec, h_gru_all, noise),
            onnx_model_fpath,
            input_names=["window", "identity", "emotion", "input_latents", "noise"],
            output_names=["prediction", "output_latents"],
            opset_version=14,
            do_constant_folding=False,  # TODO Setting to true causes gpu/cpu device mismatch error
            dynamic_axes={
                "window": [0],
                "identity": [0],  # actor_name
                "emotion": [0],
                "input_latents": [2],
                "noise": [0],
                "prediction": [0],
                "output_latents": [2],
            },
        )


class WrapModel(torch.nn.Module):
    def __init__(
        self,
        cfg_train: dict,
        cfg_dataset: dict,
        cfg_inference: dict,
        network: NetworkBaseDiffusion,
        diffusion,
        sample_fn,
    ) -> None:
        super(WrapModel, self).__init__()
        self.cfg_train = cfg_train
        self.cfg_dataset = cfg_dataset
        self.cfg_inference = cfg_inference
        self.model = network

        self.model.TIMESTEP_RESPACING = cfg_inference.TIMESTEP_RESPACING
        if self.cfg_inference.USE_DELTA_OUTPUT:
            self.model.set_mode(mode="streaming_stateless_output_delta")
        else:
            self.model.set_mode(mode="streaming_stateless")

        self.diffusion = diffusion
        self.sample_fn = sample_fn
        assert not cfg_inference.USE_DDIM  # TODO Only support ddpm for now

    def forward(
        self,
        window: torch.Tensor,
        actor_vec: torch.Tensor,
        emotion_vec: torch.Tensor,
        h_gru_all: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        num_frame = (
            self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["block_frame_size"]
            + self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["left_truncate"]
            + self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["right_truncate"]
        )
        prediction, h_gru_all = self.sample_fn(
            self.model,
            (
                window.shape[0],
                num_frame,
                self.model.skin_dim + self.model.tongue_dim + self.model.jaw_dim + self.model.eye_dim,
            ),
            clip_denoised=False,
            model_kwargs={
                "audio": window,
                "actor_vec": actor_vec,
                "emotion_vec": emotion_vec,
                "h_gru_all": h_gru_all,
            },
            skip_timesteps=self.cfg_inference.SKIP_STEPS,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=noise,
            const_noise=False,
            device="cuda",
        )
        return prediction, h_gru_all


def run(
    training_run_name_full: str,
    cfg_train_mod: dict | None = None,
    cfg_dataset_mod: dict | None = None,
    cfg_inference_mod: dict | None = None,
) -> None:
    converter = Converter(training_run_name_full, cfg_train_mod, cfg_dataset_mod, cfg_inference_mod)
    converter.setup()
    converter.convert_model()
