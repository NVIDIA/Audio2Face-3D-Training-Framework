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
import numpy as np
import librosa

from transformers import Wav2Vec2Processor

from audio2face import utils
from audio2face.config_base import config_train_diffusion_base, config_dataset_base, config_inference_diffusion_base
from audio2face.deps.motion_diffusion_model.diffusion.script_utils import create_gaussian_diffusion
import audio2face.geometry.maya_cache as maya_cache
from audio2face.dataset import AudioProvider, TargetProviderFullFace
from audio2face.dataset import AnimationSegmentDataset
from audio2face.train_diffusion import Trainer


class InferenceEngine(Trainer):
    def __init__(
        self,
        training_run_name_full: str,
        cfg_train_mod: dict | None = None,
        cfg_dataset_mod: dict | None = None,
        cfg_inference_mod: dict | None = None,
    ):
        self.training_run_name_full = training_run_name_full
        self.cfg_train_mod = cfg_train_mod
        self.cfg_dataset_mod = cfg_dataset_mod
        self.cfg_inference_mod = cfg_inference_mod

    def _get_model_kwargs(
        self,
        window: torch.Tensor,
        actor_vec: torch.Tensor,
        emotion_vec: torch.Tensor,
        h_gru_all: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Helper to create model_kwargs based on inference mode
        """
        base_kwargs = {
            "audio": window,
            "actor_vec": actor_vec,
            "emotion_vec": emotion_vec,
        }

        # Handle GRU states
        if "stateless" in self.cfg_inference.INFERENCE_MODE:
            base_kwargs["h_gru_all"] = h_gru_all

        return base_kwargs

    def _get_sample_params(
        self,
        num_frames: int,
        mean_geometry: torch.Tensor,
        noise: torch.Tensor,
    ) -> dict:
        """
        Helper to get common sample parameters
        """
        return {
            "model": self.network,
            "shape": (1, num_frames, mean_geometry.shape[1]),
            "clip_denoised": False,
            "skip_timesteps": self.cfg_inference.SKIP_STEPS,
            "init_image": None,
            "progress": True,
            "dump_steps": None,
            "noise": noise,
            "const_noise": False,
            "device": "cuda",
        }

    def setup(self) -> None:
        self.cfg_train = utils.module_to_easy_dict(config_train_diffusion_base, modifier=self.cfg_train_mod)
        self.cfg_dataset = utils.module_to_easy_dict(config_dataset_base, modifier=self.cfg_dataset_mod)
        self.cfg_inference = utils.module_to_easy_dict(config_inference_diffusion_base, modifier=self.cfg_inference_mod)
        utils.validate_identifier_or_raise(self.training_run_name_full, "Training Run Name Full")

        train_output_root = utils.get_module_var("TRAIN_OUTPUT_ROOT", config_train_diffusion_base, self.cfg_train_mod)
        self.training_artifact_dir = os.path.normpath(os.path.join(train_output_root, self.training_run_name_full))
        self.result_dir = os.path.normpath(
            os.path.join(self.cfg_inference.INFERENCE_OUTPUT_ROOT, self.training_run_name_full)
        )

        self.audio_provider = AudioProvider(self.cfg_train, self.cfg_dataset)
        self.target_provider = TargetProviderFullFace(self.cfg_train, self.cfg_dataset)

        self.dataset = AnimationSegmentDataset(
            self.cfg_train,
            self.cfg_dataset,
            self.audio_provider,
            None,
            self.target_provider,
        )

        network_info_full, network_info_deploy = self.compose_network_info()
        self.network_params = network_info_full["params"]
        self.network_extra_data = self.compose_network_extra_data()
        self.network = self.create_network()

        state_dict = torch.load(os.path.join(self.training_artifact_dir, "weights.pth"), map_location="cuda")
        self.network.load_state_dict(state_dict, strict=True)

        self.network = self.network.to(torch.device("cuda"))
        self.network.eval()
        self.network.TIMESTEP_RESPACING = self.cfg_inference.TIMESTEP_RESPACING

        self.diffusion = create_gaussian_diffusion(
            steps=self.cfg_train.DIFFUSION_STEPS,
            noise_schedule=self.cfg_train.DIFFUSION_NOISE_SCHEDULE,
            predict_xstart=True,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            timestep_respacing=self.cfg_inference.TIMESTEP_RESPACING,
        )

        self.sample_fn = (
            self.diffusion.p_sample_loop if not self.cfg_inference.USE_DDIM else self.diffusion.ddim_sample_loop
        )

    def run(self) -> None:
        # TODO Deferring mkdir as workaround - ONNX Converter depends on InferenceEngine.setup(), should be disentangled
        os.makedirs(self.result_dir, exist_ok=True)

        mean_geometry = self.target_provider.get_mean_geometry(self.cfg_inference.ACTOR_NAME)
        mean_geometry = mean_geometry.unsqueeze(0)

        audio_path = self.cfg_inference.AUDIO_PATH
        name = os.path.basename(audio_path).split(".")[0]
        audio_data, audio_rate = librosa.load(os.path.join(audio_path), sr=16000)

        if self.cfg_train.NO_STANDARDIZE_AUDIO:
            audio_data = audio_data[np.newaxis]
        else:
            processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
            audio_data = processor(
                audio_data, return_tensors="pt", padding="longest", audio_rate=audio_rate
            ).input_values
        audio_data = torch.FloatTensor(audio_data).to(device="cuda")

        actor_names = self.cfg_dataset.ACTOR_NAMES
        actor_vecs = np.eye(len(actor_names))

        iter = actor_names.index(self.cfg_inference.CONDITION)
        actor_vec = actor_vecs[iter]
        actor_vec = np.reshape(actor_vec, (-1, actor_vec.shape[0]))
        actor_vec = torch.FloatTensor(actor_vec).to(device="cuda")

        emo_len = len(
            utils.get_network_emotion_names(
                self.cfg_dataset.SHOT_EMOTION_NAMES, self.cfg_train.SHOT_EMOTION_NAME_FOR_ALL_ZEROS
            )
        )
        emotion_vec = np.zeros([1, emo_len])
        emotion_idx = self.cfg_dataset.SHOT_EMOTION_NAMES.index(self.cfg_inference.EMOTION_LABEL)
        if emotion_idx > 0:  # neutral is all 0
            emotion_vec[0, emotion_idx - 1] = 1
        emotion_vec = torch.FloatTensor(emotion_vec).to(device="cuda")

        if self.cfg_inference.INFERENCE_MODE == "offline" or self.cfg_inference.INFERENCE_MODE == "offline_onnx":
            num_frames = int(audio_data.shape[1] / audio_rate * self.cfg_train.TARGET_FPS)

            if self.cfg_inference.INFERENCE_MODE == "offline_onnx":
                self.network.set_mode(mode="offline_onnx")

            if self.cfg_inference.DETERMINISTIC_NOISE_PATH:
                noise = torch.from_numpy(np.load(self.cfg_inference.DETERMINISTIC_NOISE_PATH)).to(device="cuda")
                # noise = None
            else:
                noise = torch.randn(
                    [1, self.diffusion.num_timesteps + 1, num_frames, mean_geometry.shape[1]], device="cuda"
                )

            model_kwargs = self._get_model_kwargs(audio_data, actor_vec, emotion_vec)
            sample_params = self._get_sample_params(num_frames, mean_geometry, noise)

            # denoise the entire audio track at once
            prediction = self.sample_fn(**sample_params, model_kwargs=model_kwargs)
            prediction = prediction.squeeze()
            prediction = prediction.detach().cpu().numpy()  # nFrame (30fps), 3V shape vertices position

        elif self.cfg_inference.INFERENCE_MODE in [
            "streaming",
            "streaming_stateless",
            "streaming_stateless_output_delta",
            "streaming_stateless_onnx",
            "streaming_stateless_trt",
            "streaming_stateless_output_delta_onnx",
            "streaming_stateless_output_delta_trt",
        ]:
            if self.cfg_inference.INFERENCE_MODE == "streaming":
                self.network.set_mode(mode=self.cfg_inference.INFERENCE_MODE)
            elif self.cfg_inference.INFERENCE_MODE in ["streaming_stateless", "streaming_stateless_output_delta"]:
                self.network.set_mode(mode=self.cfg_inference.INFERENCE_MODE)
                batch_size = 1
                steps = len(self.network.timestep_idx)
                h_gru_all = torch.zeros(
                    [steps, self.network.num_gru_layers, batch_size, self.network.gru_feature_dim], device="cuda"
                )
            elif self.cfg_inference.INFERENCE_MODE in [
                "streaming_stateless_onnx",
                "streaming_stateless_output_delta_onnx",
            ]:
                import onnxruntime as ort

                onnx_model_path = "simplified.onnx"
                ort_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])
                steps = 2
                batch_size = 1
                h_gru_all = torch.zeros(
                    [steps, self.network.num_gru_layers, batch_size, self.network.gru_feature_dim], device="cuda"
                )
            elif self.cfg_inference.INFERENCE_MODE in [
                "streaming_stateless_trt",
                "streaming_stateless_output_delta_trt",
            ]:
                import tensorrt as trt

                # need this otherwise could not find InstanceNorm_TRT version 1.
                # See https://github.com/onnx/onnx-tensorrt/issues/597#issuecomment-769877649
                trt.init_libnvinfer_plugins(None, "")
                TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                steps = 2  # hardcoded for now

                def load_engine(trt_runtime, plan_path: str):
                    with open(plan_path, "rb") as f:
                        engine_data = f.read()
                    return trt_runtime.deserialize_cuda_engine(engine_data)

                runtime = trt.Runtime(TRT_LOGGER)
                engine = load_engine(runtime, "streaming_stateless_model.trt")
                context = engine.create_execution_context()

                # Allocate buffers for input and output using torch tensors
                batch_size = 1
                h_gru_all = torch.zeros(
                    [steps, self.network.num_gru_layers, batch_size, self.network.gru_feature_dim], device="cuda"
                )  # initialize gru hidden states
                output_prediction = torch.empty(
                    (
                        batch_size,
                        self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["block_frame_size"]
                        + self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["left_truncate"]
                        + self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["right_truncate"],
                        mean_geometry.shape[1],
                    ),
                    dtype=torch.float32,
                ).to(torch.device("cuda"))
                output_h_gru_all = torch.empty(
                    (steps, self.network.num_gru_layers, batch_size, self.network.gru_feature_dim), dtype=torch.float32
                ).to(torch.device("cuda"))

            audio_data = torch.concat(
                [
                    torch.zeros(
                        [1, self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["audio_pad_left"]]
                    ).cuda(),
                    audio_data,
                    torch.zeros(
                        [1, self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["audio_pad_right"]]
                    ).cuda(),
                ],
                dim=1,
            )
            window_size = self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["window_size"]
            stride = window_size // 2  # Stride is equal 15fps

            # Prepare to collect predictions
            all_predictions = []

            num_frames = int(window_size / audio_rate * self.cfg_train.TARGET_FPS)

            if self.cfg_inference.DETERMINISTIC_NOISE_PATH:
                noise = torch.from_numpy(np.load(self.cfg_inference.DETERMINISTIC_NOISE_PATH)).to(device="cuda")
                # noise = None
            else:
                noise = torch.randn(
                    [1, self.diffusion.num_timesteps + 1, num_frames, mean_geometry.shape[1]], device="cuda"
                )

            for start in range(0, audio_data.shape[1] - window_size + 1, stride):
                window = audio_data[:, start : start + window_size]

                model_kwargs = (
                    self._get_model_kwargs(window, actor_vec, emotion_vec, h_gru_all)
                    if "stateless" in self.cfg_inference.INFERENCE_MODE
                    else self._get_model_kwargs(window, actor_vec, emotion_vec)
                )
                sample_params = self._get_sample_params(num_frames, mean_geometry, noise)

                # Generate prediction for the current window
                if self.cfg_inference.INFERENCE_MODE == "streaming":
                    prediction = self.sample_fn(**sample_params, model_kwargs=model_kwargs)
                elif self.cfg_inference.INFERENCE_MODE == "streaming_stateless":
                    prediction, h_gru_all = self.sample_fn(**sample_params, model_kwargs=model_kwargs)
                elif self.cfg_inference.INFERENCE_MODE == "streaming_stateless_output_delta":
                    prediction, h_gru_all = self.sample_fn(**sample_params, model_kwargs=model_kwargs)
                elif self.cfg_inference.INFERENCE_MODE in [
                    "streaming_stateless_onnx",
                    "streaming_stateless_output_delta_onnx",
                ]:
                    inputs = {
                        "window": window.cpu().numpy(),
                        "identity": actor_vec.cpu().numpy(),  # actor_name
                        "emotion": emotion_vec.cpu().numpy(),
                        "input_latents": h_gru_all.cpu().numpy(),
                        "noise": noise.cpu().numpy(),
                    }

                    # Run inference
                    outputs = ort_session.run(None, inputs)

                    # Extract outputs
                    prediction = torch.tensor(outputs[0])
                    h_gru_all = torch.tensor(outputs[1])

                elif self.cfg_inference.INFERENCE_MODE in [
                    "streaming_stateless_trt",
                    "streaming_stateless_output_delta_trt",
                ]:
                    context.set_tensor_address("window", window.data_ptr())
                    context.set_tensor_address("identity", actor_vec.data_ptr())  # actor_name
                    context.set_tensor_address("emotion", emotion_vec.data_ptr())
                    context.set_tensor_address("input_latents", h_gru_all.data_ptr())
                    context.set_tensor_address("noise", noise.data_ptr())
                    context.set_tensor_address("output_latents", output_h_gru_all.data_ptr())
                    context.set_tensor_address("prediction", output_prediction.data_ptr())

                    bindings = [
                        int(window.data_ptr()),
                        int(actor_vec.data_ptr()),
                        int(emotion_vec.data_ptr()),
                        int(h_gru_all.data_ptr()),
                        int(noise.data_ptr()),
                        int(output_h_gru_all.data_ptr()),
                        int(output_prediction.data_ptr()),
                    ]

                    context.execute_v2(bindings=bindings)

                    prediction = output_prediction.clone()
                    h_gru_all.copy_(output_h_gru_all)

                # Extract and save the middle frame prediction
                prediction = (
                    prediction.squeeze()
                    .detach()
                    .cpu()
                    .numpy()[
                        self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["left_truncate"] : (
                            self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["left_truncate"]
                            + self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["block_frame_size"]
                        )
                    ]
                )
                frame_to_append = (prediction).copy()  # Take the middle frame of the window
                all_predictions.append(frame_to_append)
            prediction = np.concatenate(all_predictions, axis=0)[
                int(
                    self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["audio_pad_left"]
                    / audio_rate
                    * self.cfg_train.TARGET_FPS
                )
                - self.cfg_train.STREAMING_CFG[str(int(self.cfg_train.TARGET_FPS))]["left_truncate"] :
            ]  # nFrame, 3V
        self.out_file_name = (
            name
            + "_"
            + self.cfg_inference.ACTOR_NAME
            + "_"
            + self.cfg_inference.CONDITION
            + "_"
            + self.cfg_inference.EMOTION_LABEL
            + "_"
            + self.cfg_inference.INFERENCE_MODE
        )

        # Save NPY file if enabled
        if self.cfg_inference.OUTPUT_NPY_FILE:
            np.save(os.path.join(self.result_dir, self.out_file_name), prediction)

        if (
            self.cfg_inference.INFERENCE_MODE == "streaming_stateless_output_delta"
            or self.cfg_inference.INFERENCE_MODE == "streaming_stateless_output_delta_onnx"
            or self.cfg_inference.INFERENCE_MODE == "streaming_stateless_output_delta_trt"
        ):
            offset, scale = (
                self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["translate_skin"],
                self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["scale_skin"],
            )
            tongue_data_scale, tongue_data_offset, jaw_data_scale = (
                self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["scale_tongue"],
                self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["translate_tongue"],
                self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["scale_jaw"],
            )
            # Export maya cache if enabled
            if self.cfg_inference.OUTPUT_MAYA_CACHE:
                maya_cache.export_animation(
                    prediction[:, : self.network.skin_dim].reshape(-1, self.network.skin_dim // 3, 3)
                    + mean_geometry[: self.network.skin_dim].reshape(-1, 3) * scale
                    + offset,
                    self.cfg_train.TARGET_FPS,
                    os.path.join(self.result_dir, self.out_file_name + "_delta"),
                )  # animation should be NXVX3

                pred_tongue_mesh = (
                    prediction[:, self.network.skin_dim : (self.network.skin_dim + self.network.tongue_dim)].reshape(
                        -1, self.network.tongue_dim // 3, 3
                    )
                    + mean_geometry[self.network.skin_dim : (self.network.skin_dim + self.network.tongue_dim)].reshape(
                        -1, 3
                    )
                    * tongue_data_scale
                    + tongue_data_offset
                )
                maya_cache.export_animation(
                    pred_tongue_mesh,
                    self.cfg_train.TARGET_FPS,
                    os.path.join(self.result_dir, self.out_file_name + "_tongue_delta"),
                )

            # Save additional NPY files if enabled
            if self.cfg_inference.OUTPUT_NPY_FILE:
                np.save(
                    os.path.join(self.result_dir, self.out_file_name + "_jaw"),
                    prediction[
                        :,
                        (self.network.skin_dim + self.network.tongue_dim) : (
                            self.network.skin_dim + self.network.tongue_dim + self.network.jaw_dim
                        ),
                    ],
                )
                np.save(
                    os.path.join(self.result_dir, self.out_file_name + "_eye"),
                    prediction[:, -self.network.eye_dim :],
                )
        else:
            offset, scale = (
                self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["translate_skin"],
                self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["scale_skin"],
            )
            tongue_data_scale, tongue_data_offset, jaw_data_scale = (
                self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["scale_tongue"],
                self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["translate_tongue"],
                self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["scale_jaw"],
            )
            # Export maya cache if enabled
            if self.cfg_inference.OUTPUT_MAYA_CACHE:
                maya_cache.export_animation(
                    (prediction[:, : self.network.skin_dim].reshape(-1, self.network.skin_dim // 3, 3) * scale)
                    + offset,
                    self.cfg_train.TARGET_FPS,
                    os.path.join(self.result_dir, self.out_file_name),
                )  # animation should be NXVX3
                pred_tongue_mesh = (
                    prediction[:, self.network.skin_dim : (self.network.skin_dim + self.network.tongue_dim)].reshape(
                        -1, self.network.tongue_dim // 3, 3
                    )
                    * tongue_data_scale
                ) + tongue_data_offset
                maya_cache.export_animation(
                    pred_tongue_mesh,
                    self.cfg_train.TARGET_FPS,
                    os.path.join(self.result_dir, self.out_file_name + "_tongue"),
                )
            # Save additional NPY files if enabled
            if self.cfg_inference.OUTPUT_NPY_FILE:
                np.save(
                    os.path.join(self.result_dir, self.out_file_name + "_jaw"),
                    jaw_data_scale
                    * prediction[
                        :,
                        (self.network.skin_dim + self.network.tongue_dim) : (
                            self.network.skin_dim + self.network.tongue_dim + self.network.jaw_dim
                        ),
                    ],
                )
                np.save(
                    os.path.join(self.result_dir, self.out_file_name + "_eye"),
                    self.target_provider.data_info[self.cfg_inference.ACTOR_NAME]["scale_eye"]
                    * prediction[:, -self.network.eye_dim :],
                )


def run(
    training_run_name_full: str,
    cfg_train_mod: dict | None = None,
    cfg_dataset_mod: dict | None = None,
    cfg_inference_mod: dict | None = None,
) -> None:
    inference_engine = InferenceEngine(training_run_name_full, cfg_train_mod, cfg_dataset_mod, cfg_inference_mod)
    inference_engine.setup()
    inference_engine.run()
