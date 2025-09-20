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
import datetime
import importlib
import shutil
import json
import time
import logging

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from audio2face import layers, losses, utils
from audio2face.dataset import AnimationSequenceDataset
from audio2face.dataset import AudioProvider, PhonemeProvider, TargetProviderFullFace
from audio2face.config_base import config_dataset_base, config_train_regression_base
from audio2face.emotion import ImplicitEmotionManager
from audio2face.networks.base import NetworkBaseRegression

FRAMEWORK_ROOT_DIR = utils.get_framework_root_dir()


class Trainer:
    def __init__(
        self,
        cfg_train_mod: dict | None = None,
        cfg_dataset_mod: dict | None = None,
    ) -> None:
        self.cfg_train_mod = cfg_train_mod
        self.cfg_dataset_mod = cfg_dataset_mod

    def gen_run_name_full(self) -> str:
        run_name = utils.get_module_var("RUN_NAME", config_train_regression_base, self.cfg_train_mod)
        utils.validate_identifier_or_raise(run_name, "Train RUN_NAME")
        return datetime.datetime.today().strftime("%y%m%d_%H%M%S_") + run_name

    def prepare_out_dirs(self) -> None:
        train_output_root = utils.get_module_var("TRAIN_OUTPUT_ROOT", config_train_regression_base, self.cfg_train_mod)
        self.out_dir = os.path.normpath(os.path.join(train_output_root, self.run_name_full))
        self.deploy_dir = os.path.normpath(os.path.join(self.out_dir, "deploy"))
        self.configs_dir = os.path.normpath(os.path.join(self.out_dir, "configs"))
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.deploy_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)

    def compose_network_params_from_dataset(self) -> dict:
        network_emotion_names = utils.get_network_emotion_names(
            self.cfg_dataset.SHOT_EMOTION_NAMES, self.cfg_train.SHOT_EMOTION_NAME_FOR_ALL_ZEROS
        )
        network_params_from_dataset = {
            "explicit_emotions": network_emotion_names,
            "num_shapes_skin": self.target_provider.channel_size["skin_coeffs"],
            "num_verts_skin": self.target_provider.get_unique_data_info_value("num_verts_skin"),
            "num_shapes_tongue": self.target_provider.channel_size["tongue_coeffs"],
            "num_verts_tongue": self.target_provider.get_unique_data_info_value("num_verts_tongue"),
            "result_jaw_size": self.target_provider.channel_size["jaw"],
            "result_eyes_size": self.target_provider.channel_size["eye"],
        }
        if self.phoneme_provider is not None:
            network_params_from_dataset["num_phonemes"] = self.phoneme_provider.phoneme_detector.num_phonemes
        return network_params_from_dataset

    def compose_network_info(self) -> tuple[dict, dict]:
        network_params_from_dataset = self.compose_network_params_from_dataset()
        default_emotion = self.dataset.emotion_name2vec(self.cfg_train.DEFAULT_SHOT_EMOTION_NAME).tolist()
        network_id_info = {
            "type": self.cfg_train.NETWORK_TYPE,
            "actor": self.cfg_dataset.ACTOR_NAMES[0] if len(self.cfg_dataset.ACTOR_NAMES) == 1 else "multi",
            "version": self.cfg_train.NETWORK_VERSION,
            "output": "geometry",
        }
        network_info_full = {
            "id": network_id_info,
            "name": self.cfg_train.NETWORK_NAME,
            "params": self.cfg_train.NETWORK_HYPER_PARAMS | network_params_from_dataset,
            "audio_params": self.cfg_train.AUDIO_PARAMS,
            "default_emotion": default_emotion,
        }
        network_info_deploy = {
            "id": network_id_info,
            "params": {
                "implicit_emotion_len": self.cfg_train.NETWORK_HYPER_PARAMS["implicit_emotion_len"],
                "explicit_emotions": network_params_from_dataset["explicit_emotions"],
                "default_emotion": default_emotion,
                "identities": self.cfg_dataset.ACTOR_NAMES,
                "num_shapes_skin": network_params_from_dataset["num_shapes_skin"],
                "num_verts_skin": network_params_from_dataset["num_verts_skin"],
                "num_shapes_tongue": network_params_from_dataset["num_shapes_tongue"],
                "num_verts_tongue": network_params_from_dataset["num_verts_tongue"],
                "result_jaw_size": network_params_from_dataset["result_jaw_size"],
                "result_eyes_size": network_params_from_dataset["result_eyes_size"],
            },
            "audio_params": self.cfg_train.AUDIO_PARAMS,
        }
        return network_info_full, network_info_deploy

    def export_framework_version(self) -> None:
        shutil.copy(os.path.join(FRAMEWORK_ROOT_DIR, "VERSION.md"), self.configs_dir)

    def export_run_info(self) -> None:
        with open(os.path.join(self.configs_dir, "info.txt"), "w") as f:
            f.write(self.cfg_train.RUN_INFO)

    def export_preproc_info(self) -> None:
        preproc_info = {"preproc_run_name_full": self.cfg_train.PREPROC_RUN_NAME_FULL}
        utils.json_dump_pretty(preproc_info, os.path.join(self.configs_dir, "preproc_info.json"))

    def export_cfg_mod(self) -> None:
        if self.cfg_train_mod is not None:
            utils.json_dump_pretty(self.cfg_train_mod, os.path.join(self.configs_dir, "config_train_modifier.json"))
        if self.cfg_dataset_mod is not None:
            utils.json_dump_pretty(self.cfg_dataset_mod, os.path.join(self.configs_dir, "config_dataset_modifier.json"))

    def export_cfg_full(self) -> None:
        utils.json_dump_pretty(self.cfg_train, os.path.join(self.configs_dir, "config_train_full.json"))
        utils.json_dump_pretty(self.cfg_dataset, os.path.join(self.configs_dir, "config_dataset_full.json"))

    def export_network_info_deploy(self, network_info_deploy: dict) -> None:
        utils.json_dump_pretty(network_info_deploy, os.path.join(self.deploy_dir, "network_info.json"))

    def export_network_info_full(self, network_info_full: dict) -> None:
        utils.json_dump_pretty(network_info_full, os.path.join(self.out_dir, "network_info_full.json"))

    def export_trt_info(self, network_info_deploy: dict) -> None:
        audio_len = network_info_deploy["audio_params"]["buffer_len"]
        emo_len = network_info_deploy["params"]["implicit_emotion_len"] + len(
            network_info_deploy["params"]["explicit_emotions"]
        )
        trt_info = {
            "trt_build_param": {
                "cuda_in_graphics": ["--memPoolSize=tacticSharedMem:0.046875"],
                "batch": [
                    f"--minShapes=input:1x1x{audio_len},emotion:1x1x{emo_len}",
                    f"--maxShapes=input:{{MAX_BATCH_SIZE}}x1x{audio_len},emotion:{{MAX_BATCH_SIZE}}x1x{emo_len}",
                    f"--optShapes=input:{{OPT_BATCH_SIZE}}x1x{audio_len},emotion:{{OPT_BATCH_SIZE}}x1x{emo_len}",
                ],
            },
        }
        utils.json_dump_pretty(trt_info, os.path.join(self.deploy_dir, "trt_info.json"))

    def export_state(self) -> None:
        utils.json_dump_pretty(utils.get_state_info(FRAMEWORK_ROOT_DIR), os.path.join(self.configs_dir, "state.json"))

    def export_meta_data(self, network_info_full: dict, network_info_deploy: dict) -> None:
        self.export_framework_version()
        self.export_run_info()
        self.export_preproc_info()
        self.export_cfg_mod()
        self.export_network_info_deploy(network_info_deploy)
        self.export_trt_info(network_info_deploy)
        if not utils.is_partial_exposure():
            self.export_cfg_full()
            self.export_network_info_full(network_info_full)
            self.export_state()

    def setup_torch(self) -> None:
        torch.hub.set_dir(self.cfg_train.TORCH_CACHE_ROOT)
        if self.cfg_train.REPRODUCIBLE:
            torch.manual_seed(self.cfg_train.RNG_SEED_TORCH)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        logging.info(f"Using GPU [{device_id}] {device_name}")

    def create_network(self) -> NetworkBaseRegression:
        network_py_module_name = "audio2face.networks." + self.cfg_train.NETWORK_NAME
        Network = importlib.import_module(network_py_module_name).Network
        network = Network(self.network_params)
        return network

    def read_lip_dist_verts(self, actor_name: str) -> dict | None:
        if self.cfg_dataset.SKIN_LIP_DIST_VERTEX_LIST_FPATH.get(actor_name) is None:
            logging.warning(f'SKIN_LIP_DIST_VERTEX_LIST_FPATH is not provided for actor "{actor_name}" in the config')
            return None
        with open(self.cfg_dataset.SKIN_LIP_DIST_VERTEX_LIST_FPATH.get(actor_name)) as f:
            lip_dist_verts = json.load(f)
        return lip_dist_verts

    def read_lip_size_verts(self, actor_name: str) -> dict | None:
        if self.cfg_dataset.SKIN_LIP_SIZE_VERTEX_LIST_FPATH.get(actor_name) is None:
            logging.warning(f'SKIN_LIP_SIZE_VERTEX_LIST_FPATH is not provided for actor "{actor_name}" in the config')
            return None
        with open(self.cfg_dataset.SKIN_LIP_SIZE_VERTEX_LIST_FPATH.get(actor_name)) as f:
            lip_size_verts = json.load(f)
        return lip_size_verts

    def read_eye_dist_verts(self, actor_name: str) -> dict | None:
        if self.cfg_dataset.SKIN_EYE_DIST_VERTEX_LIST_FPATH.get(actor_name) is None:
            logging.warning(f'SKIN_EYE_DIST_VERTEX_LIST_FPATH is not provided for actor "{actor_name}" in the config')
            return None
        with open(self.cfg_dataset.SKIN_EYE_DIST_VERTEX_LIST_FPATH.get(actor_name)) as f:
            eye_dist_verts = json.load(f)
        return eye_dist_verts

    def using_phonemes(self) -> bool:
        return len(self.cfg_train.PHONEME_FORCING_LANGS) > 0 and self.cfg_train.LOSS_PHONEME_ALPHA is not None

    def gen_phoneme_weights(self) -> list[torch.Tensor]:
        phonemes_weights_all = []
        for lang in self.phoneme_provider.phoneme_detector.langs:
            num_phonemes = self.phoneme_provider.phoneme_detector.num_phonemes[lang]
            sil_token_idx = self.phoneme_provider.phoneme_detector.get_sil_token_idx(lang)
            phoneme_weights = [1.0] * num_phonemes
            phoneme_weights[sil_token_idx] = self.cfg_train.LOSS_PHONEME_SIL_WEIGHT
            phoneme_weights = torch.tensor(phoneme_weights).float().cuda()
            phonemes_weights_all.append(phoneme_weights)
        return phonemes_weights_all

    def gen_sample_weights(self, samples: list[str], weight_map: dict[str, float] | None) -> torch.Tensor | None:
        if weight_map is None:
            return None
        sample_weights = [weight_map.get(key, 1.0) for key in samples]
        return torch.tensor(sample_weights).float().cuda()

    def export_implicit_emo_db(self) -> None:
        implicit_emo_manager = ImplicitEmotionManager(
            emo_db=self.implicit_emo_db.W.detach().cpu().numpy(),
            emo_specs={shot.id: (shot.start_global, shot.len) for shot in self.dataset.shots.values()},
        )
        if self.cfg_train.COMPACTIFY_IMPLICIT_EMO_DB:
            logging.info("Compactifying Implicit Emotion Database")
            implicit_emo_manager.compactify(self.dataset.clips)
        implicit_emo_manager.save_npz(os.path.join(self.deploy_dir, "implicit_emo_db.npz"))
        if not utils.is_partial_exposure():
            implicit_emo_manager.save_pkl(os.path.join(self.out_dir, "implicit_emo_db.pkl"))

    def export_onnx(self) -> None:
        input_tensor_name = "input"
        emotion_tensor_name = "emotion"
        result_tensor_name = "result"
        joint_emotion_len = self.network_params["implicit_emotion_len"] + len(self.network_params["explicit_emotions"])
        dummy_input = (
            torch.zeros((1, 1, self.cfg_train.AUDIO_PARAMS["buffer_len"]), dtype=torch.float32).cuda(),
            torch.zeros((1, 1, joint_emotion_len), dtype=torch.float32).cuda(),
        )
        self.network.set_mode("onnx")
        torch.onnx.export(
            self.network,
            dummy_input,
            os.path.join(self.deploy_dir, "network.onnx"),
            verbose=False,
            input_names=[input_tensor_name, emotion_tensor_name],
            output_names=[result_tensor_name],
            opset_version=14,
            dynamic_axes={
                input_tensor_name: {0: "batch_size"},
                emotion_tensor_name: {0: "batch_size"},
                result_tensor_name: {0: "batch_size"},
            },
        )

    def train_step(self, epoch: int, it: int, batch: utils.EasyDict[torch.Tensor | list]) -> None:
        batch.x = batch.x.cuda()
        batch.y = batch.y.cuda()
        batch.explicit_emo = batch.explicit_emo.cuda()
        batch.x_norm_factor = batch.x_norm_factor.cuda()
        if hasattr(batch, "phonemes"):
            batch.phonemes = batch.phonemes.cuda()

        self.optimizer.zero_grad()

        implicit_emo = self.implicit_emo_db(batch.global_frame_idx)
        joint_emo = torch.cat((implicit_emo, batch.explicit_emo), dim=2)
        y_pred, aux_pred = self.network(batch.x, joint_emo, batch.x_norm_factor)

        using_skin_pose_losses = (
            self.cfg_train.LOSS_LIP_DIST_ALPHA is not None and self.lip_dist_verts is not None
        ) or (self.cfg_train.LOSS_LIP_SIZE_ALPHA is not None and self.lip_size_verts is not None)
        using_phoneme_losses = (
            hasattr(batch, "phonemes")
            and hasattr(aux_pred, "phonemes")
            and (self.cfg_train.LOSS_PHONEME_ALPHA is not None or self.cfg_train.LOSS_PHONEME_MOTION_ALPHA is not None)
        )

        if self.cfg_train.REMOVE_CLOSING_EYE and self.eye_dist_verts is not None:
            y_skin_pose = self.coeffs_to_pose(batch.y[..., : self.network_params["num_shapes_skin"]])
            eye_open_mask = layers.EyeOpenMask(y_skin_pose, self.eye_dist_verts, self.cfg_train.EYE_DIST_THRESHOLD)
            y_pred = y_pred * eye_open_mask.view(-1, 1, 1)
            batch.y = batch.y * eye_open_mask.view(-1, 1, 1)

        loss = 0
        loss_exp = 1.0 - self.global_step / (self.num_batches * self.cfg_train.NUM_EPOCHS)

        if self.cfg_train.LOSS_MSE_ALPHA is not None:
            loss_mse = losses.mse_loss(y_pred, batch.y)
            loss += self.cfg_train.LOSS_MSE_ALPHA * self.normloss_mse.normalize(loss_mse) ** loss_exp

        if self.cfg_train.LOSS_MOTION_ALPHA is not None:
            loss_motion = losses.motion_loss(y_pred, batch.y)
            loss += self.cfg_train.LOSS_MOTION_ALPHA * self.normloss_motion.normalize(loss_motion) ** loss_exp

        if self.cfg_train.LOSS_EMO_REG_ALPHA is not None:
            loss_emo_reg = losses.motion_reg_loss(implicit_emo)
            loss += self.cfg_train.LOSS_EMO_REG_ALPHA * self.normloss_emo_reg.normalize(loss_emo_reg) ** loss_exp

        if using_skin_pose_losses:
            y_skin_pose_pred = self.coeffs_to_pose(y_pred[..., : self.network_params["num_shapes_skin"]])
            y_skin_pose = self.coeffs_to_pose(batch.y[..., : self.network_params["num_shapes_skin"]])

            if self.cfg_train.LOSS_LIP_DIST_ALPHA is not None and self.lip_dist_verts is not None:
                loss_lip_dist = losses.lip_dist_loss(
                    y_skin_pose_pred,
                    y_skin_pose,
                    self.cfg_train.LOSS_LIP_DIST_EXP,
                    self.lip_dist_verts,
                    self.gen_sample_weights(batch.explicit_emo_name, self.cfg_train.LOSS_LIP_DIST_EMO_WEIGHTS),
                )
                loss += self.cfg_train.LOSS_LIP_DIST_ALPHA * loss_lip_dist

            if self.cfg_train.LOSS_LIP_SIZE_ALPHA is not None and self.lip_size_verts is not None:
                loss_lip_size = losses.lip_size_loss(
                    y_skin_pose_pred,
                    y_skin_pose,
                    self.lip_size_verts,
                    self.gen_sample_weights(batch.explicit_emo_name, self.cfg_train.LOSS_LIP_SIZE_EMO_WEIGHTS),
                )
                loss += self.cfg_train.LOSS_LIP_SIZE_ALPHA * loss_lip_size

        if self.cfg_train.LOSS_VOL_STAB_REG_ALPHA is not None:
            loss_vol_stab_reg = losses.vol_stab_reg_loss(batch.x, y_pred, self.cfg_train.LOSS_VOL_STAB_REG_EXP)
            loss += self.cfg_train.LOSS_VOL_STAB_REG_ALPHA * loss_vol_stab_reg

        if using_phoneme_losses:
            # Filtering phoneme data by language
            phonemes_pred_all = []
            phonemes_all = []
            for lang_idx, lang in enumerate(self.phoneme_provider.phoneme_detector.langs):
                num_phonemes = self.phoneme_provider.phoneme_detector.num_phonemes[lang]
                lang_mask = [sample_lang == lang for sample_lang in batch.lang]
                phonemes_pred_all.append(aux_pred.phonemes[lang_idx][lang_mask, ...])
                phonemes_all.append(batch.phonemes[lang_mask, ..., :num_phonemes])  # Trimming the padding

            if self.cfg_train.LOSS_PHONEME_ALPHA is not None:
                loss_phoneme = losses.phoneme_loss(phonemes_pred_all, phonemes_all, self.phonemes_weights_all)
                loss += self.cfg_train.LOSS_PHONEME_ALPHA * self.normloss_phoneme.normalize(loss_phoneme) ** loss_exp

            if self.cfg_train.LOSS_PHONEME_MOTION_ALPHA is not None:
                loss_phoneme_motion = losses.phoneme_motion_loss(phonemes_pred_all, phonemes_all)
                loss += (
                    self.cfg_train.LOSS_PHONEME_MOTION_ALPHA
                    * self.normloss_phoneme_motion.normalize(loss_phoneme_motion) ** loss_exp
                )

        loss.backward()
        self.optimizer.step()

        if it % self.cfg_train.LOG_PERIOD == 0:
            loss_value = loss.item()
            r2_value = losses.r2(y_pred, batch.y).item()

            self.writer.add_scalar("loss", loss_value, self.global_step)
            self.writer.add_scalar("r2", r2_value, self.global_step)

            if not utils.is_partial_exposure():
                self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
                if self.cfg_train.LOSS_MSE_ALPHA is not None:
                    self.writer.add_scalar("losses/mse", loss_mse.item(), self.global_step)
                if self.cfg_train.LOSS_MOTION_ALPHA is not None:
                    self.writer.add_scalar("losses/motion", loss_motion.item(), self.global_step)
                if self.cfg_train.LOSS_EMO_REG_ALPHA is not None:
                    self.writer.add_scalar("losses/emo_reg", loss_emo_reg.item(), self.global_step)
                if self.cfg_train.LOSS_LIP_DIST_ALPHA is not None and self.lip_dist_verts is not None:
                    self.writer.add_scalar("losses/lip_dist", loss_lip_dist.item(), self.global_step)
                if self.cfg_train.LOSS_LIP_SIZE_ALPHA is not None and self.lip_size_verts is not None:
                    self.writer.add_scalar("losses/lip_size", loss_lip_size.item(), self.global_step)
                if self.cfg_train.LOSS_VOL_STAB_REG_ALPHA is not None:
                    self.writer.add_scalar("losses/vol_stab_reg", loss_vol_stab_reg.item(), self.global_step)
                if self.cfg_train.LOSS_PHONEME_ALPHA is not None and using_phoneme_losses:
                    self.writer.add_scalar("losses/phoneme", loss_phoneme.item(), self.global_step)
                if self.cfg_train.LOSS_PHONEME_MOTION_ALPHA is not None and using_phoneme_losses:
                    self.writer.add_scalar("losses/phoneme_motion", loss_phoneme_motion.item(), self.global_step)

                self.network.eval()
                self.implicit_emo_db.eval()
                joint_emo_zero = torch.cat((torch.zeros_like(implicit_emo), batch.explicit_emo), dim=2)
                y_pred_emo_zero, _ = self.network(batch.x, joint_emo_zero, batch.x_norm_factor)
                r2_emo_zero_value = losses.r2(y_pred_emo_zero, batch.y).item()
                self.network.train()
                self.implicit_emo_db.train()
                self.writer.add_scalar("r2_emo_zero", r2_emo_zero_value, self.global_step)

            time_remain = (time.time() - self.time_start) * (
                float(self.num_batches * self.cfg_train.NUM_EPOCHS) / float(self.global_step + 1) - 1.0
            )
            time_remain_str = str(datetime.timedelta(seconds=round(time_remain)))
            logging.info(
                "Epoch: {:3d}/{} | Iter: {:4d}/{} | Loss = {:8.5f} | R2 = {:7.4f} | Remaining time: {}".format(
                    epoch, self.cfg_train.NUM_EPOCHS, it, self.num_batches, loss_value, r2_value, time_remain_str
                )
            )

        self.global_step += 1

    def setup(self) -> None:
        self.run_name_full = self.gen_run_name_full()
        self.prepare_out_dirs()

        utils.setup_logging(os.path.join(self.out_dir, "log.log"))
        logging.info("--------------------------------------------------------------------------------")
        logging.info(f"Training experiment: {self.run_name_full}")
        logging.info("--------------------------------------------------------------------------------")

        utils.validate_cfg_mod(self.cfg_train_mod, config_train_regression_base, "train")
        utils.validate_cfg_mod(self.cfg_dataset_mod, config_dataset_base, "dataset")
        self.cfg_train = utils.module_to_easy_dict(config_train_regression_base, modifier=self.cfg_train_mod)
        self.cfg_dataset = utils.module_to_easy_dict(config_dataset_base, modifier=self.cfg_dataset_mod)
        for actor_name, preproc_run_name_full in self.cfg_train.PREPROC_RUN_NAME_FULL.items():
            if preproc_run_name_full.startswith("XXXXXX_XXXXXX"):
                raise ValueError(f'Please update PREPROC_RUN_NAME_FULL in config_train.py for actor "{actor_name}"')
            utils.validate_identifier_or_raise(preproc_run_name_full, f'PREPROC_RUN_NAME_FULL for actor "{actor_name}"')

        self.setup_torch()

        self.audio_provider = AudioProvider(self.cfg_train, self.cfg_dataset)
        self.phoneme_provider = PhonemeProvider(self.cfg_train, self.cfg_dataset) if self.using_phonemes() else None
        self.target_provider = TargetProviderFullFace(self.cfg_train, self.cfg_dataset)
        self.dataset = AnimationSequenceDataset(
            self.cfg_train,
            self.cfg_dataset,
            self.audio_provider,
            self.phoneme_provider,
            self.target_provider,
        )
        self.dataset_loader = torch.utils.data.DataLoader(self.dataset, self.cfg_train.BATCH_SIZE, shuffle=True)
        self.num_batches = len(self.dataset_loader)

        network_info_full, network_info_deploy = self.compose_network_info()
        self.export_meta_data(network_info_full, network_info_deploy)
        self.network_params = network_info_full["params"]

        self.network = self.create_network()
        self.network = self.network.cuda()
        self.network.set_mode("train")
        if self.cfg_train.PRETRAINED_NET_FPATH is not None:
            logging.info(f"Using pre-trained weights: {self.cfg_train.PRETRAINED_NET_FPATH}")
            self.network.load_state_dict(torch.load(self.cfg_train.PRETRAINED_NET_FPATH))
        else:
            logging.info("Training the network from scratch")

        self.implicit_emo_db = layers.ImplicitEmotionDB(
            self.dataset.total_shot_frames, self.network_params["implicit_emotion_len"], self.cfg_train.EMO_INIT_SIGMA
        )
        self.implicit_emo_db = self.implicit_emo_db.cuda()

        # TODO Implement support for multiple actors (where currently actor_name_0 is used)
        if len(self.cfg_dataset.ACTOR_NAMES) > 1:
            raise NotImplementedError("Train regression: only single actor in ACTOR_NAMES is supported")
        actor_name_0 = self.cfg_dataset.ACTOR_NAMES[0]

        skin_pca_shapes = self.target_provider.get_skin_pca_shapes(actor_name_0)
        self.coeffs_to_pose = layers.CoeffsToPose(skin_pca_shapes["shapes_matrix"], skin_pca_shapes["shapes_mean"])
        self.coeffs_to_pose = self.coeffs_to_pose.cuda()

        optimizer_param_groups = [
            {
                "params": self.network.parameters(),
                "lr": self.cfg_train.LEARNING_RATE,
            },
            {
                "params": self.implicit_emo_db.parameters(),
                "lr": self.cfg_train.LEARNING_RATE * self.cfg_train.EMO_LR_MULT,
            },
        ]
        self.optimizer = optim.Adam(optimizer_param_groups, lr=self.cfg_train.LEARNING_RATE)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.cfg_train.LR_STEP_GAMMA)
        self.normloss_mse = losses.LossNormalizer("stddev", 0.99)
        self.normloss_motion = losses.LossNormalizer("stddev", 0.99)
        self.normloss_emo_reg = losses.LossNormalizer("stddev", 0.99)
        self.normloss_phoneme = losses.LossNormalizer("stddev", 0.99)
        self.normloss_phoneme_motion = losses.LossNormalizer("stddev", 0.99)

        if self.cfg_train.LOSS_LIP_DIST_ALPHA is not None:
            self.lip_dist_verts = self.read_lip_dist_verts(actor_name_0)
        if self.cfg_train.LOSS_LIP_SIZE_ALPHA is not None:
            self.lip_size_verts = self.read_lip_size_verts(actor_name_0)
        if self.cfg_train.REMOVE_CLOSING_EYE:
            self.eye_dist_verts = self.read_eye_dist_verts(actor_name_0)
        if self.using_phonemes():
            self.phonemes_weights_all = self.gen_phoneme_weights()

        self.writer = SummaryWriter(self.out_dir)
        logging.info(f"Output directory: {self.out_dir}")
        logging.info("--------------------------------------------------------------------------------")

        self.network.train()
        self.implicit_emo_db.train()

        self.global_step = 0

    def preload_dataset(self) -> None:
        self.dataset.preload()

    def run(self) -> dict:
        self.time_start = time.time()

        try:
            for epoch in range(self.cfg_train.NUM_EPOCHS):
                for it, batch in enumerate(self.dataset_loader):
                    self.train_step(epoch, it, batch)
                self.lr_scheduler.step()
                # TODO add checkpoint save every K epochs
        except KeyboardInterrupt:
            logging.info("Training interrupted")

        training_time = time.time() - self.time_start
        training_time_str = str(datetime.timedelta(seconds=round(training_time)))
        logging.info("--------------------------------------------------------------------------------")
        logging.info(f"Total training time: {training_time_str} ({training_time:.2f} sec)")

        self.export_implicit_emo_db()
        self.export_onnx()
        if not utils.is_partial_exposure():
            torch.save(self.network.state_dict(), os.path.join(self.out_dir, "weights.pth"))

        logging.info("--------------------------------------------------------------------------------")
        logging.info(f"Training artifacts: {self.out_dir}")
        logging.info(f"Deployment artifacts: {self.deploy_dir}")
        logging.info("--------------------------------------------------------------------------------")
        logging.info(f"Mapping to local FS: /framework is {os.getenv('EXTERNAL_A2F_FRAMEWORK_ROOT') or '/framework'}")
        logging.info(f"Mapping to local FS: /datasets is {os.getenv('EXTERNAL_A2F_DATASETS_ROOT') or '/datasets'}")
        logging.info(f"Mapping to local FS: /workspace is {os.getenv('EXTERNAL_A2F_WORKSPACE_ROOT') or '/workspace'}")
        logging.info("--------------------------------------------------------------------------------")
        logging.info(f"Training Run Name Full: {self.run_name_full}")
        logging.info("--------------------------------------------------------------------------------")

        return {
            "training_time": training_time,
            "run_name_full": self.run_name_full,
            "out_dir": self.out_dir,
            "deploy_dir": self.deploy_dir,
        }


def run(
    cfg_train_mod: dict | None = None,
    cfg_dataset_mod: dict | None = None,
) -> dict:
    trainer = Trainer(cfg_train_mod, cfg_dataset_mod)
    trainer.setup()
    trainer.preload_dataset()
    result = trainer.run()
    return result
