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
import torch
import torch.nn as nn
import torchaudio

from audio2face import utils
from audio2face import layers
from audio2face.networks.base import NetworkBaseRegression


# JoinApplySplit
# Join Batch/Seq -> Apply -> Split Batch/Seq
def JAS(f, x: torch.Tensor) -> torch.Tensor:
    batch, seq_len = x.shape[:2]
    x = x.view([batch * seq_len] + list(x.shape)[2:])
    x = f(x)
    x = x.view([batch, seq_len] + list(x.shape)[1:])
    return x


class Network(NetworkBaseRegression):
    def __init__(self, params: dict) -> None:
        super(Network, self).__init__()

        self.use_w2v_features = params.get("use_w2v_features", False)
        self.use_autocorr_features = params.get("use_autocorr_features", False)
        if not any([self.use_w2v_features, self.use_autocorr_features]):
            raise ValueError("Network: no audio features are being used")

        self.num_phonemes = params.get("num_phonemes", None)
        self.use_phoneme_forcing = self.num_phonemes is not None and self.use_w2v_features

        self.implicit_emo_len = params.get("implicit_emotion_len", 16)
        self.explicit_emo_len = len(params.get("explicit_emotions", []))
        self.explicit_emo_emb_len = params.get("explicit_emo_emb_len", 8)
        self.joint_emo_emb_len = self.implicit_emo_len + self.explicit_emo_emb_len

        self.num_shapes_skin = params.get("num_shapes_skin", 0)
        self.num_shapes_tongue = params.get("num_shapes_tongue", 0)
        self.result_jaw_size = params.get("result_jaw_size", 0)
        self.result_eyes_size = params.get("result_eyes_size", 0)
        self.result_skin_jaw_eyes_size = self.num_shapes_skin + self.result_jaw_size + self.result_eyes_size

        self.relu = nn.ReLU(inplace=True)
        self.drop = layers.GDropout()

        self.joint_audio_features_len = 0

        if self.use_w2v_features:
            self.w2v_num_layers = params.get("w2v_num_layers", 1)
            self.w2v_freeze = params.get("w2v_freeze", True)
            self.w2v_x_scale = 1.0
            self.w2v = torchaudio.pipelines.WAV2VEC2_BASE.get_model().cuda()
            self.w2v.encoder.transformer.layers = self.w2v.encoder.transformer.layers[: self.w2v_num_layers]
            if self.w2v_freeze:
                for param in self.w2v.parameters():
                    param.requires_grad = False
            self.w2v_feature_coeffs = nn.Parameter(torch.ones(self.w2v_num_layers), requires_grad=True)
            self.w2v_linear_mapping = nn.Linear(768, 256)
            self.joint_audio_features_len += 256

        if self.use_autocorr_features:
            self.autocorr_params = params.get("autocorr_params", {})
            self.autocorr_x_scale = 8.912028121636435  # derived from input data std() from an old legacy dataset
            self.autocorr = layers.AutoCorr(**self.autocorr_params)
            self.freq1 = nn.Conv2d(1, 72, (1, 3), stride=(1, 2), padding=(0, 1))
            self.freq2 = nn.Conv2d(72, 108, (1, 3), stride=(1, 2), padding=(0, 1))
            self.freq3 = nn.Conv2d(108, 162, (1, 3), stride=(1, 2), padding=(0, 1))
            self.freq4 = nn.Conv2d(162, 243, (1, 3), stride=(1, 2), padding=(0, 1))
            self.freq5 = nn.Conv2d(243, 256, (1, 2), stride=(1, 1), padding=(0, 0))
            self.joint_audio_features_len += 256

        if self.use_phoneme_forcing:
            self.phoneme_conv1 = nn.Conv2d(256, 256, (3, 1), stride=(1, 1), padding=(1, 0))
            self.phoneme_conv2 = nn.Conv2d(256, 256, (3, 1), stride=(1, 1), padding=(1, 0))
            self.phoneme_regressors = nn.ModuleList()
            for lang in self.num_phonemes.keys():
                self.phoneme_regressors.append(nn.Linear(256, self.num_phonemes[lang]))

        self.time1 = nn.Conv2d(self.joint_audio_features_len, 256, (3, 1), stride=(2, 1), padding=(1, 0))
        self.time2 = nn.Conv2d(256 + self.joint_emo_emb_len, 256, (3, 1), stride=(2, 1), padding=(1, 0))
        self.time3 = nn.Conv2d(256 + self.joint_emo_emb_len, 256, (3, 1), stride=(2, 1), padding=(1, 0))
        self.time4 = nn.Conv2d(256 + self.joint_emo_emb_len, 256, (3, 1), stride=(2, 1), padding=(1, 0))
        self.time5 = nn.Linear(2 * (256 + self.joint_emo_emb_len), 256)

        self.linear_skin_jaw_eyes1 = nn.Linear(256 + self.joint_emo_emb_len, self.result_skin_jaw_eyes_size)
        self.linear_skin_jaw_eyes2 = nn.Linear(self.result_skin_jaw_eyes_size, self.result_skin_jaw_eyes_size)

        self.linear_tongue1 = nn.Linear(256 + self.joint_emo_emb_len, self.num_shapes_tongue)
        self.linear_tongue2 = nn.Linear(self.num_shapes_tongue, self.num_shapes_tongue)

        self.linear_explicit_emo = nn.Linear(self.explicit_emo_len, self.explicit_emo_emb_len)

    def w2v_features_weighted(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.w2v.extract_features(x)
        features = torch.stack(features, dim=-1)
        features = (self.w2v_feature_coeffs * features).sum(axis=-1) / self.w2v_feature_coeffs.sum()
        return features

    def w2v_audio_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.w2v_x_scale
        x = JAS(self.w2v_features_weighted, x)  # [batch, seq, 25, 768]
        x = self.relu(self.w2v_linear_mapping(x))  # [batch, seq, 25, 256]
        x = x.permute(0, 1, 3, 2)[..., None]  # [batch, seq, 256, 25, 1]
        return x

    def autocorr_audio_features(self, x: torch.Tensor, x_norm_factor: torch.Tensor | None = None) -> torch.Tensor:
        if x_norm_factor is not None:
            x = x / x_norm_factor.view([-1, 1, 1])
        x = x * self.autocorr_x_scale
        x = self.drop(self.autocorr(x))
        x = self.drop(layers.NormRelu(JAS(self.freq1, x)))
        x = self.drop(layers.NormRelu(JAS(self.freq2, x)))
        x = self.drop(layers.NormRelu(JAS(self.freq3, x)))
        x = self.drop(layers.NormRelu(JAS(self.freq4, x)))
        x = self.drop(layers.NormRelu(JAS(self.freq5, x)))
        return x

    def forward(
        self,
        x: torch.Tensor,
        emotion: torch.Tensor,
        x_norm_factor: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, utils.EasyDict]:
        out_aux = {}

        implicit_emo = emotion[:, :, : self.implicit_emo_len]
        explicit_emo = emotion[:, :, self.implicit_emo_len :]
        explicit_emo_emb = self.linear_explicit_emo(explicit_emo)

        audio_features = []
        if self.use_w2v_features:
            x_w2v = self.w2v_audio_features(x)
            audio_features.append(x_w2v)
        if self.use_autocorr_features:
            x_autocorr = self.autocorr_audio_features(x, x_norm_factor)
            audio_features.append(x_autocorr)
        x = torch.cat(audio_features, dim=-3)

        if self.use_phoneme_forcing and self.mode == "train":
            x_phonemes = self.drop(layers.NormRelu(JAS(self.phoneme_conv1, x_w2v)))
            x_phonemes = self.drop(layers.NormRelu(JAS(self.phoneme_conv2, x_phonemes)))
            out_phonemes = []
            for phoneme_regressor in self.phoneme_regressors:
                out_phonemes.append(phoneme_regressor(x_phonemes[..., 0].transpose(2, 3)))
            out_aux["phonemes"] = out_phonemes

        x = self.drop(layers.NormRelu(JAS(self.time1, x)))
        x = layers.ConcatEmotion(x, implicit_emo)
        x = layers.ConcatEmotion(x, explicit_emo_emb)
        x = self.drop(layers.NormRelu(JAS(self.time2, x)))
        x = layers.ConcatEmotion(x, implicit_emo)
        x = layers.ConcatEmotion(x, explicit_emo_emb)
        x = self.drop(layers.NormRelu(JAS(self.time3, x)))
        x = layers.ConcatEmotion(x, implicit_emo)
        x = layers.ConcatEmotion(x, explicit_emo_emb)
        x = self.drop(layers.NormRelu(JAS(self.time4, x)))
        x = layers.ConcatEmotion(x, implicit_emo)
        x = layers.ConcatEmotion(x, explicit_emo_emb)

        x = x.view(x.shape[0], x.shape[1], -1)

        x = layers.NormRelu(JAS(self.time5, x))
        x = layers.ConcatEmotion(x, implicit_emo)
        x = layers.ConcatEmotion(x, explicit_emo_emb)

        out_skin_jaw_eyes = JAS(self.linear_skin_jaw_eyes1, x)
        out_skin_jaw_eyes = JAS(self.linear_skin_jaw_eyes2, out_skin_jaw_eyes)

        out_tongue = JAS(self.linear_tongue1, x)
        out_tongue = JAS(self.linear_tongue2, out_tongue)

        out = torch.cat(
            (
                out_skin_jaw_eyes[:, :, : self.num_shapes_skin],
                out_tongue,
                out_skin_jaw_eyes[:, :, self.num_shapes_skin :],
            ),
            dim=2,
        )

        if self.mode == "train":
            return out, utils.EasyDict(out_aux)
        elif self.mode == "onnx":
            return out
