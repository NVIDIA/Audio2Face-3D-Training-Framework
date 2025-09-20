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
from collections import defaultdict
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
from transformers.models.hubert.modeling_hubert import HubertModel, _compute_mask_indices

from audio2face import utils
from audio2face.deps.motion_diffusion_model.diffusion.respace import space_timesteps
from audio2face.layers import linear1d
from audio2face.networks.base import NetworkBaseDiffusion


def _init_final_layer(layer: nn.Sequential) -> None:
    nn.init.constant_(layer[-1].weight, 0)
    nn.init.constant_(layer[-1].bias, 0)


def _create_final_layer(input_dim: int, output_dim: int, ratio: float = 1.0) -> nn.Sequential:
    intermediate_dim = int(input_dim * ratio)
    return nn.Sequential(nn.Linear(input_dim, intermediate_dim), nn.ReLU(), nn.Linear(intermediate_dim, output_dim))


class CustomHubertModel(HubertModel):
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.expand(
                mask_time_indices.sum(), -1
            )  # For use deterministic algorithms
            # hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states


def interp_input(
    audio_feature: torch.Tensor,
    vertices: torch.Tensor,
    hubert_fps: int,
    output_anim_fps: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Aligns 50 FPS audio embeddings with either 30 FPS or 60 FPS animation data
    using frame rate specific resampling strategies.
    """
    # Validate supported frame rates
    B = audio_feature.shape[0]
    if hubert_fps != 50 or output_anim_fps not in {30, 60}:
        raise ValueError("Only supports 50->30/60 FPS conversion")
    # Get target animation sequence length
    num_frames = vertices.shape[1]
    if output_anim_fps == 60:
        audio_feature = linear1d(
            audio_feature.transpose(1, 2), out_length=num_frames, align_corners=True, half_pixel_centers=False
        ).transpose(1, 2)
    elif output_anim_fps == 30:
        audio_feature = linear1d(
            audio_feature.transpose(1, 2), out_length=int(num_frames * 2), align_corners=True, half_pixel_centers=False
        ).transpose(1, 2)

    audio_feature = torch.reshape(
        audio_feature,
        (B, num_frames, -1),
    )
    return audio_feature, vertices, num_frames


class EmbedGeometry(nn.Module):
    def __init__(
        self,
        skin_dim: int,
        feature_dim: int = 256,
        tongue_dim: int = 10,
        tongue_latent_dim: int = 32,
        jaw_dim: int = 15,
        jaw_latent_dim: int = 15,
        eye_dim: int = 4,
        eye_latent_dim: int = 4,
    ) -> None:
        super(EmbedGeometry, self).__init__()
        self.skin_dim = skin_dim
        self.tongue_dim = tongue_dim
        self.jaw_dim = jaw_dim
        self.eye_dim = eye_dim
        self.skin_embedding = nn.Sequential(
            nn.Linear(skin_dim, feature_dim * 2),
            nn.Conv1d(1, 1, kernel_size=9, padding=4),
            nn.ReLU(),
        )

        self.tongue_embedding = nn.Sequential(
            nn.Linear(tongue_dim, tongue_latent_dim * 2),
            nn.Conv1d(1, 1, kernel_size=9, padding=4),
            nn.ReLU(),
        )

        self.jaw_embedding = nn.Sequential(
            nn.Linear(jaw_dim, jaw_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(jaw_latent_dim * 2, jaw_latent_dim * 2),
            nn.ReLU(),
        )

        self.eye_embedding = nn.Sequential(
            nn.Linear(eye_dim, eye_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(eye_latent_dim * 2, eye_latent_dim * 2),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Linear(
                (feature_dim + tongue_latent_dim + jaw_latent_dim + eye_latent_dim) * 2,
                (feature_dim + tongue_latent_dim + jaw_latent_dim + eye_latent_dim) * 2,
            ),
            nn.ReLU(),
            nn.Linear(
                (feature_dim + tongue_latent_dim + jaw_latent_dim + eye_latent_dim) * 2,
                (feature_dim + tongue_latent_dim + jaw_latent_dim + eye_latent_dim),
            ),
        )

    def forward(self, geom_input):
        skin_data = geom_input[:, :, : self.skin_dim]
        tongue_data = geom_input[:, :, self.skin_dim : self.skin_dim + self.tongue_dim]
        jaw_data = geom_input[:, :, self.skin_dim + self.tongue_dim : self.skin_dim + self.tongue_dim + self.jaw_dim]
        eye_data = geom_input[:, :, self.skin_dim + self.tongue_dim + self.jaw_dim :]

        seq, B, V = skin_data.shape
        skin_data = skin_data.reshape(-1, 1, V)  # reshape for support with batch size > 1
        skin_data_out = self.skin_embedding(skin_data)
        skin_data_out = skin_data_out.reshape(seq, B, -1)

        seq, B, V = tongue_data.shape
        tongue_data = tongue_data.reshape(-1, 1, V)  # reshape for support with batch size > 1
        tongue_data_out = self.tongue_embedding(tongue_data)
        tongue_data_out = tongue_data_out.reshape(seq, B, -1)
        jaw_data_out = self.jaw_embedding(jaw_data)
        eye_data_out = self.eye_embedding(eye_data)

        final_out = torch.cat((skin_data_out, tongue_data_out, jaw_data_out, eye_data_out), dim=2)
        final_out = self.final(final_out)
        return final_out


class Network(NetworkBaseDiffusion):
    def __init__(self, params: dict, extra_data: dict) -> None:
        super().__init__()
        self.params = params

        diffusion_steps = self.params["DIFFUSION_STEPS"]
        self.hubert_fps = utils.HUBERT_FEATURE_FPS
        self.output_anim_fps = self.params["TARGET_FPS"]
        self.onehot_diffsteps = torch.FloatTensor(torch.eye(diffusion_steps)).cuda()

        self.num_actors = len(self.params["actor_names"])
        self.emo_len = len(self.params["explicit_emotions"])
        emo_embedding_dim = self.params["NETWORK_HYPER_PARAMS"]["emo_embedding_dim"]
        actor_embedding_dim = self.params["NETWORK_HYPER_PARAMS"]["actor_embedding_dim"]

        self.skin_dim = self.params["num_verts_skin"] * 3
        self.tongue_dim = self.params["num_verts_tongue"] * 3
        self.jaw_dim = self.params["result_jaw_size"]
        self.eye_dim = self.params["result_eyes_size"]

        feature_dim = self.params["NETWORK_HYPER_PARAMS"]["feature_dim"]
        jaw_latent_dim = self.params["NETWORK_HYPER_PARAMS"]["jaw_latent_dim"]
        tongue_latent_dim = self.params["NETWORK_HYPER_PARAMS"]["tongue_latent_dim"]
        eye_latent_dim = self.params["NETWORK_HYPER_PARAMS"]["eye_latent_dim"]
        gru_feature_dim = self.params["NETWORK_HYPER_PARAMS"]["gru_feature_dim"]
        num_gru_layers = self.params["NETWORK_HYPER_PARAMS"]["num_gru_layers"]

        # Audio Encoder
        self.audio_model = CustomHubertModel.from_pretrained("facebook/hubert-base-ls960")  # TODO use TORCH_CACHE_ROOT
        self.device = torch.device("cuda")
        self.audio_model.feature_extractor._freeze_parameters()
        for name, param in self.audio_model.named_parameters():
            if (
                name.startswith("feature_projection")
                or name.startswith("encoder.layers.0.")
                or name.startswith("encoder.layers.1.")
            ):
                param.requires_grad = False

        # conditional projection
        self.proj_hubert = nn.Linear(self.params["NETWORK_HYPER_PARAMS"]["hubert_feature_dim"], feature_dim)

        self.geometry_embedding = EmbedGeometry(
            self.skin_dim,
            feature_dim,
            self.tongue_dim,
            tongue_latent_dim,
            self.jaw_dim,
            jaw_latent_dim,
            self.eye_dim,
            eye_latent_dim,
        )

        # timestep projection
        self.proj_time = nn.Sequential(
            nn.Linear(diffusion_steps, feature_dim),
            nn.Mish(),
        )

        # facial decoder
        self.gru = nn.GRU(
            feature_dim * 3
            + emo_embedding_dim
            + actor_embedding_dim
            + tongue_latent_dim
            + jaw_latent_dim
            + eye_latent_dim,
            gru_feature_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.final_layer_skin = _create_final_layer(gru_feature_dim, self.skin_dim, ratio=2)
        self.final_layer_tongue = _create_final_layer(gru_feature_dim, self.tongue_dim, ratio=2)
        self.final_layer_jaw = _create_final_layer(gru_feature_dim, self.jaw_dim, ratio=0.5)
        self.final_layer_eye = _create_final_layer(gru_feature_dim, self.eye_dim, ratio=0.5)

        for layer in [self.final_layer_skin, self.final_layer_tongue, self.final_layer_jaw, self.final_layer_eye]:
            _init_final_layer(layer)

        self.embed_actor = nn.Linear(self.num_actors, actor_embedding_dim, bias=False)
        self.embed_emo_vector = nn.Linear(self.emo_len, emo_embedding_dim, bias=False)

        self.num_gru_layers = num_gru_layers
        self.gru_feature_dim = gru_feature_dim
        self.params = params
        self.data_info = extra_data["data_info"]
        self.mean_geometry = extra_data["mean_geometry"]
        self.actor_names = self.params["actor_names"]

        self.scale = torch.tensor(
            [self.data_info[actor_name]["scale_skin"] for actor_name in self.actor_names],
            dtype=torch.float32,
            device="cuda",
        )  # nID
        self.offset = torch.tensor(
            [self.data_info[actor_name]["translate_skin"] for actor_name in self.actor_names],
            dtype=torch.float32,
            device="cuda",
        )  # nID x 3
        self.tongue_data_scale = torch.tensor(
            [self.data_info[actor_name]["scale_tongue"] for actor_name in self.actor_names],
            dtype=torch.float32,
            device="cuda",
        )  # nID
        self.tongue_data_offset = torch.tensor(
            [self.data_info[actor_name]["translate_tongue"] for actor_name in self.actor_names],
            dtype=torch.float32,
            device="cuda",
        )  # nID x 3
        self.jaw_data_scale = torch.tensor(
            [self.data_info[actor_name]["scale_jaw"] for actor_name in self.actor_names],
            dtype=torch.float32,
            device="cuda",
        )  # nID
        self.eyeball_data_scale = torch.tensor(
            [self.data_info[actor_name]["scale_eye"] for actor_name in self.actor_names],
            dtype=torch.float32,
            device="cuda",
        )  # nID
        self.mean_geometry = torch.stack([self.mean_geometry[actor_name] for actor_name in self.actor_names], dim=0).to(
            dtype=torch.float32, device="cuda"
        )  # nID

        self.streaming_cfg = self.params["STREAMING_CFG"][str(int(self.output_anim_fps))]

    def _process_inputs(self, geom_input, audio):
        """
        Common input processing for all forward methods
        """
        # Project to latent space
        geom_input = geom_input.permute(1, 0, 2)
        geom_input = self.geometry_embedding(geom_input)
        geom_input = geom_input.permute(1, 0, 2)

        # Process audio embeddings
        hidden_states = self.audio_model(audio).last_hidden_state
        hidden_states, geom_input, frame_num = interp_input(
            hidden_states, geom_input, self.hubert_fps, self.output_anim_fps
        )
        return hidden_states[:, :frame_num], geom_input[:, :frame_num], frame_num

    def _prepare_conditioning_tokens(
        self, audio: torch.Tensor, t: torch.Tensor, frame_num: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create conditioning tokens used in all forward variants
        """
        hubert_tokens = self.proj_hubert(audio)
        t_tokens = self.proj_time(t).repeat(frame_num, 1, 1).permute(1, 0, 2)
        return hubert_tokens, t_tokens

    def _generate_output(self, output: torch.Tensor, mean_geometry: torch.Tensor) -> torch.Tensor:
        """
        Common output generation for all forward methods
        """
        output_skin = self.final_layer_skin(output)
        output_tongue = self.final_layer_tongue(output)
        output_jaw = self.final_layer_jaw(output)
        output_eye = self.final_layer_eye(output)
        return torch.cat([output_skin, output_tongue, output_jaw, output_eye], dim=-1) + mean_geometry

    def set_mode(self, mode="streaming"):
        if mode == "streaming":
            self.inference_mode = "streaming"
            self.forward = self.forward_streaming
            self.reset_hidden_state_dict()  # reset hidden state before inference.
        elif mode == "streaming_stateless" or mode == "streaming_stateless_output_delta":
            self.inference_mode = mode
            self.forward = self.forward_streaming_stateless

            steps = self.params["DIFFUSION_STEPS"]
            timestep_respacing = self.TIMESTEP_RESPACING
            if not timestep_respacing:
                timestep_respacing = [steps]
            self.timestep_idx = torch.tensor(
                np.sort(np.array(list(space_timesteps(self.params["DIFFUSION_STEPS"], timestep_respacing))))
            ).cuda()  # used for selecting the proper input hidden state of gru

            # self.reset_hidden_state_dict() # No need of hidden state dict since they are passed in/out via forward.
        elif mode == "offline":
            self.inference_mode = "offline"
            self.forward = self.forward_offline
        elif mode == "offline_onnx":
            import onnxruntime as ort

            self.inference_mode = "offline_onnx"
            self.forward = self.forward_offline_onnx
            onnx_model_path = "offline.onnx"
            self.ort_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

    def default_hidden_states(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(
            [self.num_gru_layers, batch_size, self.gru_feature_dim], device="cuda"
        )  # assume sample size is 1

    def reset_hidden_state_dict(self) -> None:  # call this before inference a new track to clear the old hidden states.
        self.h = defaultdict(self.default_hidden_states)

    def forward_offline(
        self,
        geom_input: torch.Tensor,
        t: torch.Tensor,
        audio: torch.Tensor,
        actor_vec: torch.Tensor,
        emotion_vec: torch.Tensor,
    ) -> torch.Tensor:
        t = self.onehot_diffsteps[t]

        mean_geometry = actor_vec @ self.mean_geometry
        mean_geometry = mean_geometry.unsqueeze(1)  # 1,1, 3V
        actor_embedding = self.embed_actor(actor_vec)  # 1, 256
        emo_embedding = self.embed_emo_vector(emotion_vec)  # 1, emo_embedding_dim

        audio, geom_input, frame_num = self._process_inputs(geom_input, audio)
        hubert_tokens, t_tokens = self._prepare_conditioning_tokens(audio, t, frame_num)

        emo_embedding = emo_embedding.unsqueeze(1).repeat(1, frame_num, 1)  # 1, T, emo_embedding_dim
        actor_embedding = actor_embedding.unsqueeze(1).repeat(1, frame_num, 1)  # 1, T, emo_embedding_dim
        combined_tokens = torch.cat([hubert_tokens, geom_input, actor_embedding, t_tokens, emo_embedding], dim=-1)

        output, _ = self.gru(combined_tokens)  # combined_tokens: [1, 103, 829], output [1, 103, 256])

        output = self._generate_output(output, mean_geometry)
        return output  # 1, T(30fps), 3V

    def forward_offline_onnx(
        self,
        geom_input: torch.Tensor,
        t: torch.Tensor,
        audio: torch.Tensor,
        actor_vec: torch.Tensor,
        emotion_vec: torch.Tensor,
    ) -> torch.Tensor:
        geom_input_np = geom_input.cpu().numpy()
        t_np = t.cpu().numpy()
        audio_np = audio.cpu().numpy()
        actor_vec_np = actor_vec.cpu().numpy()
        emotion_vec_np = emotion_vec.cpu().numpy()

        ort_inputs = {
            "geom_input": geom_input_np,
            "t": t_np,
            "audio": audio_np,
            "actor_vec": actor_vec_np,  # Unused, excluded from final onnx export despite inclusion here.
            "emotion_vec": emotion_vec_np,
        }

        ort_outs = self.ort_session.run(None, ort_inputs)
        output = torch.tensor(ort_outs[0], dtype=torch.float32).to(geom_input.device)
        return output

    def forward_streaming(
        self,
        geom_input: torch.Tensor,
        t: torch.Tensor,
        audio: torch.Tensor,
        actor_vec: torch.Tensor,
        emotion_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        geom_input: B, 30, nVertices
        t: 1, -> Make it B,
        audio: B, 16000
        emotion_vec: 1, 11
        """
        if not torch.jit.is_tracing():
            assert not self.training
            assert audio.shape[1] == self.streaming_cfg["window_size"]
            assert t.shape == torch.Size([1])

        # streaming related configs
        left_truncate = self.streaming_cfg["left_truncate"]
        right_truncate = self.streaming_cfg["right_truncate"]
        block_frame_size = self.streaming_cfg["block_frame_size"]

        t_scalar = t[0].item()
        t = self.onehot_diffsteps[t]

        mean_geometry = actor_vec @ self.mean_geometry
        mean_geometry = mean_geometry.unsqueeze(1)  # 1,1, 3V
        actor_embedding = self.embed_actor(actor_vec)  # 1, 256
        emo_embedding = self.embed_emo_vector(emotion_vec)  # 1, emo_embedding_dim -> B, emo_embedding_dim

        audio, geom_input, frame_num = self._process_inputs(geom_input, audio)
        hubert_tokens, t_tokens = self._prepare_conditioning_tokens(audio, t, frame_num)

        emo_embedding = emo_embedding.unsqueeze(1).repeat(1, frame_num, 1)  # 1, T, emo_embedding_dim
        actor_embedding = actor_embedding.unsqueeze(1).repeat(1, frame_num, 1)  # 1, T, actor_embedding_dim

        combined_tokens = torch.cat(
            [
                hubert_tokens[:, left_truncate : left_truncate + block_frame_size],
                geom_input[:, left_truncate : left_truncate + block_frame_size],
                actor_embedding[:, left_truncate : left_truncate + block_frame_size],
                t_tokens[:, left_truncate : left_truncate + block_frame_size],
                emo_embedding[:, left_truncate : left_truncate + block_frame_size],
            ],
            dim=-1,
        )
        output, h = self.gru(combined_tokens, self.h[t_scalar])  # combined_tokens: [1, 103, 768], output [1, 103, 256])

        self.h[t_scalar] = h.detach()  # update the hidden state for diffusion step t.

        output = self._generate_output(output, mean_geometry)

        output = torch.concat(
            [
                torch.zeros([1, left_truncate, output.shape[2]], device="cuda"),
                output,
                torch.zeros([1, right_truncate, output.shape[2]], device="cuda"),
            ],
            dim=1,
        )  # dirty trick.

        return output  # 1, T(30fps), 3V

    def forward_streaming_stateless(
        self,
        geom_input: torch.Tensor,
        t: torch.Tensor,
        audio: torch.Tensor,
        actor_vec: torch.Tensor,
        emotion_vec: torch.Tensor,
        h_gru_all: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Above forward_streaming should work fine with batch, but using hidden states cached in model.
        # This is not possible in onnx model, so need to pass in and out the hidden states as arg.
        """
        geom_input: B, 30, nVertices
        t: B, They should all have same t,i.e., we assume each batch is denoising at the same timestep t.
        audio: B, 16000
        emotion_vec: B, 11
        h_gru_all: [diffusion_steps, num_gru_layers, batch_size, self.gru_feature_dim]
        """
        if not torch.jit.is_tracing():
            assert not self.training
            assert audio.shape[1] == self.streaming_cfg["window_size"]
            assert torch.all(t == t[0])  # we assume each batch is denoising at the same timestep t.

        # streaming related configs
        left_truncate = self.streaming_cfg["left_truncate"]
        right_truncate = self.streaming_cfg["right_truncate"]
        block_frame_size = self.streaming_cfg["block_frame_size"]

        B = geom_input.shape[0]
        diffusion_t_idx = torch.nonzero(self.timestep_idx == t[0])[0, 0]

        h_gru = h_gru_all[diffusion_t_idx]  # select out the h_gru at the right diffusion step
        t = self.onehot_diffsteps[t]

        mean_geometry = actor_vec @ self.mean_geometry
        mean_geometry = mean_geometry.unsqueeze(1)  # 1,1, 3V
        actor_embedding = self.embed_actor(actor_vec)  # 1, 256
        emo_embedding = self.embed_emo_vector(emotion_vec)  # 1, emo_embedding_dim -> B, emo_embedding_dim

        audio, geom_input, frame_num = self._process_inputs(geom_input, audio)
        hubert_tokens, t_tokens = self._prepare_conditioning_tokens(audio, t, frame_num)

        actor_embedding = actor_embedding.unsqueeze(1).repeat(1, frame_num, 1)  # 1, T, emo_embedding_dim
        if emo_embedding.ndim == 2:
            emo_embedding = emo_embedding.unsqueeze(1).repeat(
                1, frame_num, 1
            )  # 1, T, emo_embedding_dim -> B, T, emo_embedding_dim
            combined_tokens = torch.cat(
                [
                    hubert_tokens[:, left_truncate : left_truncate + block_frame_size],
                    geom_input[:, left_truncate : left_truncate + block_frame_size],
                    actor_embedding[:, left_truncate : left_truncate + block_frame_size],
                    t_tokens[:, left_truncate : left_truncate + block_frame_size],
                    emo_embedding[:, left_truncate : left_truncate + block_frame_size],
                ],
                dim=-1,
            )
        else:
            if not torch.jit.is_tracing():
                assert (emo_embedding.ndim == 3) and (
                    emo_embedding.shape[1] == block_frame_size
                )  # frame-wise emo label, of shape block_frame_size, emo_embedding
            combined_tokens = torch.cat(
                [
                    hubert_tokens[:, left_truncate : left_truncate + block_frame_size],
                    geom_input[:, left_truncate : left_truncate + block_frame_size],
                    actor_embedding[:, left_truncate : left_truncate + block_frame_size],
                    t_tokens[:, left_truncate : left_truncate + block_frame_size],
                    emo_embedding,
                ],
                dim=-1,
            )

        output, h_gru = self.gru(combined_tokens, h_gru)  # combined_tokens: [B, 103, 768], output [B, 103, 256])
        h_gru_all[diffusion_t_idx] = h_gru

        output = self._generate_output(output, mean_geometry)

        output = torch.concat(
            [
                torch.zeros([B, left_truncate, output.shape[2]], device="cuda"),
                output,
                torch.zeros([B, right_truncate, output.shape[2]], device="cuda"),
            ],
            dim=1,
        )  # dirty trick.

        return output, h_gru_all  # B, T(30fps), 3V
