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
import random
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from timm.scheduler import CosineLRScheduler

from audio2face import losses, utils
from audio2face.dataset import AnimationSegmentDataset, extract_random_subsegment
from audio2face.dataset import AudioProvider, TargetProviderFullFace
from audio2face.config_base import config_dataset_base, config_train_diffusion_base, config_inference_diffusion_base
from audio2face.networks.base import NetworkBaseDiffusion

from audio2face.deps.motion_diffusion_model.diffusion.script_utils import create_gaussian_diffusion
from audio2face.deps.motion_diffusion_model.diffusion.resample import create_named_schedule_sampler

FRAMEWORK_ROOT_DIR = utils.get_framework_root_dir()


class Trainer:
    def __init__(
        self,
        cfg_train_mod: dict | None = None,
        cfg_dataset_mod: dict | None = None,
        cfg_inference_mod: dict | None = None,
    ) -> None:
        self.cfg_train_mod = cfg_train_mod
        self.cfg_dataset_mod = cfg_dataset_mod
        self.cfg_inference_mod = cfg_inference_mod

    def gen_run_name_full(self) -> str:
        run_name = utils.get_module_var("RUN_NAME", config_train_diffusion_base, self.cfg_train_mod)
        utils.validate_identifier_or_raise(run_name, "Train RUN_NAME")
        return datetime.datetime.today().strftime("%y%m%d_%H%M%S_") + run_name

    def prepare_out_dirs(self) -> None:
        train_output_root = utils.get_module_var("TRAIN_OUTPUT_ROOT", config_train_diffusion_base, self.cfg_train_mod)
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
            "actor_names": self.cfg_dataset.ACTOR_NAMES,
            "explicit_emotions": network_emotion_names,
            "num_shapes_skin": self.target_provider.get_unique_data_info_value("num_shapes_skin"),
            "num_verts_skin": self.target_provider.get_unique_data_info_value("num_verts_skin"),
            "num_shapes_tongue": self.target_provider.get_unique_data_info_value("num_shapes_tongue"),
            "num_verts_tongue": self.target_provider.get_unique_data_info_value("num_verts_tongue"),
            "result_jaw_size": self.target_provider.channel_size["jaw"],
            "result_eyes_size": self.target_provider.channel_size["eye"],
        }
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
            "params": self.cfg_train | network_params_from_dataset,
            "audio_params": self.cfg_train.AUDIO_PARAMS,
            "default_emotion": default_emotion,
        }
        target_fps = str(int(self.cfg_train.TARGET_FPS))
        network_info_deploy = {
            "id": network_id_info,
            "params": {
                "emotions": network_params_from_dataset["explicit_emotions"],
                "default_emotion": default_emotion,
                "identities": self.cfg_dataset.ACTOR_NAMES,
                "skin_size": network_params_from_dataset["num_verts_skin"] * 3,
                "tongue_size": network_params_from_dataset["num_verts_tongue"] * 3,
                "jaw_size": network_params_from_dataset["result_jaw_size"],
                "eyes_size": network_params_from_dataset["result_eyes_size"],
                "num_diffusion_steps": self.cfg_inference.NUM_DIFFUSION_STEPS,
                "num_gru_layers": self.cfg_train.NETWORK_HYPER_PARAMS["num_gru_layers"],
                "gru_latent_dim": self.cfg_train.NETWORK_HYPER_PARAMS["gru_feature_dim"],
                "num_frames_left_truncate": self.cfg_train.STREAMING_CFG[target_fps]["left_truncate"],
                "num_frames_right_truncate": self.cfg_train.STREAMING_CFG[target_fps]["right_truncate"],
                "num_frames_center": self.cfg_train.STREAMING_CFG[target_fps]["block_frame_size"],
            },
            "audio_params": self.cfg_train.AUDIO_PARAMS,
        }
        return network_info_full, network_info_deploy

    def compose_network_extra_data(self) -> dict:
        network_extra_data = {
            "data_info": self.target_provider.data_info,
            "mean_geometry": {
                actor_name: self.target_provider.get_mean_geometry(actor_name)
                for actor_name in self.cfg_dataset.ACTOR_NAMES
            },
        }
        return network_extra_data

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
        num_actors = len(network_info_deploy["params"]["identities"])
        num_emotions = len(network_info_deploy["params"]["emotions"])
        output_len = (
            network_info_deploy["params"]["skin_size"]
            + network_info_deploy["params"]["tongue_size"]
            + network_info_deploy["params"]["jaw_size"]
            + network_info_deploy["params"]["eyes_size"]
        )
        gru_latent_dim = network_info_deploy["params"]["gru_latent_dim"]
        block_size = network_info_deploy["params"]["num_frames_center"]
        num_diff_steps = network_info_deploy["params"]["num_diffusion_steps"]
        num_gru_layers = network_info_deploy["params"]["num_gru_layers"]
        left_truncate = network_info_deploy["params"]["num_frames_left_truncate"]
        right_truncate = network_info_deploy["params"]["num_frames_right_truncate"]
        trt_info = {
            "estimated_trt_builder_time": 150,
            "trt_build_param": {
                "cuda_in_graphics": ["--memPoolSize=tacticSharedMem:0.046875"],
                "batch": [
                    (
                        f"--minShapes=window:1x{audio_len},"
                        f"identity:1x{num_actors},"
                        f"emotion:1x{block_size}x{num_emotions},"
                        f"input_latents:{num_diff_steps}x{num_gru_layers}x1x{gru_latent_dim},"
                        f"noise:1x{num_diff_steps+1}x{block_size+left_truncate+right_truncate}x{output_len}"
                    ),
                    (
                        f"--maxShapes=window:{{MAX_BATCH_SIZE}}x{audio_len},"
                        f"identity:{{MAX_BATCH_SIZE}}x{num_actors},"
                        f"emotion:{{MAX_BATCH_SIZE}}x{block_size}x{num_emotions},"
                        f"input_latents:{num_diff_steps}x{num_gru_layers}x{{MAX_BATCH_SIZE}}x{gru_latent_dim},"
                        f"noise:{{MAX_BATCH_SIZE}}x{num_diff_steps+1}x{block_size+left_truncate+right_truncate}x{output_len}"
                    ),
                    (
                        f"--optShapes=window:{{OPT_BATCH_SIZE}}x{audio_len},"
                        f"identity:{{OPT_BATCH_SIZE}}x{num_actors},"
                        f"emotion:{{OPT_BATCH_SIZE}}x{block_size}x{num_emotions},"
                        f"input_latents:{num_diff_steps}x{num_gru_layers}x{{OPT_BATCH_SIZE}}x{gru_latent_dim},"
                        f"noise:{{OPT_BATCH_SIZE}}x{num_diff_steps+1}x{block_size+left_truncate+right_truncate}x{output_len}"
                    ),
                ],
            },
            "defaults": {"MAX_BATCH_SIZE": 8, "OPT_BATCH_SIZE": 3},
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
            np.random.seed(self.cfg_train.RNG_SEED_TORCH)
            random.seed(self.cfg_train.RNG_SEED_TORCH)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        logging.info(f"Using GPU [{device_id}] {device_name}")

    def create_network(self) -> NetworkBaseDiffusion:
        network_py_module_name = "audio2face.networks." + self.cfg_train.NETWORK_NAME
        Network = importlib.import_module(network_py_module_name).Network
        network = Network(self.network_params, self.network_extra_data)
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

    def gen_sample_weights(self, samples: list[str], weight_map: dict[str, float] | None) -> torch.Tensor | None:
        if weight_map is None:
            return None
        sample_weights = [weight_map.get(key, 1.0) for key in samples]
        return torch.tensor(sample_weights).float().cuda()

    def export_onnx(self) -> None:
        from audio2face import convert_onnx

        convert_onnx.run(
            training_run_name_full=self.run_name_full,
            cfg_train_mod=self.cfg_train_mod,
            cfg_dataset_mod=self.cfg_dataset_mod,
            cfg_inference_mod=self.cfg_inference_mod,
        )
        logging.info(f"ONNX model exported to: {self.run_name_full}")

    def compute_additional_losses(
        self,
        model_output: torch.Tensor,
        vertices: torch.Tensor,
        actor_name: str,
    ) -> dict:
        # Extract PCA-related parameters
        scale = torch.tensor(self.target_provider.data_info[actor_name]["scale_skin"], dtype=torch.float32).cuda()
        offset = torch.tensor(self.target_provider.data_info[actor_name]["translate_skin"], dtype=torch.float32).cuda()

        skin_pca_shapes = self.target_provider.get_skin_pca_shapes(actor_name)
        evecs_t = skin_pca_shapes["shapes_matrix"]
        evecs_t = evecs_t.reshape(evecs_t.shape[0], -1).transpose(0, 1)
        nr_vertices = evecs_t.shape[0] // 3
        pca_mean_face = skin_pca_shapes["shapes_mean"].flatten()

        # Vertex calculations
        nframes = model_output.shape[1]
        gt_skin_vert = (
            vertices[:, :, : 3 * nr_vertices].reshape(-1, nframes, nr_vertices, 3) * scale + offset
        ).reshape(-1, nframes, nr_vertices * 3)
        pred_skin_vert = (
            model_output[:, :, : 3 * nr_vertices].reshape(-1, nframes, nr_vertices, 3) * scale + offset
        ).reshape(-1, nframes, nr_vertices * 3)

        # PCA projections
        pred_skin_pca = torch.matmul(pred_skin_vert - pca_mean_face, evecs_t)
        recon_skin_vert = torch.matmul(pred_skin_pca, evecs_t.T) + pca_mean_face

        return {
            "lip_dist": (
                losses.lip_dist_loss(
                    recon_skin_vert.view(recon_skin_vert.shape[0], recon_skin_vert.shape[1], -1, 3),
                    gt_skin_vert.view(gt_skin_vert.shape[0], gt_skin_vert.shape[1], -1, 3),
                    self.cfg_train.LOSS_LIP_DIST_EXP,
                    self.lip_dist_verts,
                )
                if self.cfg_train.LOSS_LIP_DIST_ALPHA is not None and self.lip_dist_verts is not None
                else 0
            ),
            "velocity": (
                losses.motion_loss(model_output[:, :, : 3 * nr_vertices], vertices[:, :, : 3 * nr_vertices])
                if self.cfg_train.LOSS_VELOCITY_ALPHA is not None
                else 0
            ),
            "expression_smooth": (
                losses.expression_smooth_reg(model_output[:, :, : 3 * nr_vertices], self.softmask_upperface)
                if self.cfg_train.LOSS_EXP_SMOOTH_ALPHA is not None and self.softmask_upperface is not None
                else 0
            ),
        }

    def train_step(self, epoch: int, it: int, batch: utils.EasyDict[torch.Tensor | list]) -> None:
        audio = batch.x.cuda()
        vertices = batch.y.cuda()
        emotion_vec = batch.emotion_vec.cuda()
        actor_vec = batch.actor_vec.cuda()

        if self.cfg_train.TRAIN_ON_RANDOM_SUBSEGMENT:
            vertices, audio = extract_random_subsegment(
                vertices,
                audio,
                sample_rate=16000,
                fps=self.cfg_train.TARGET_FPS,
                min_frame=self.cfg_train.SUBSEGMENT_MIN_FRAME,
                max_frame=self.cfg_train.SUBSEGMENT_MAX_FRAME,
            )

        t, _ = self.schedule_sampler.sample(1, torch.device("cuda"))

        output = self.diffusion.training_losses(
            self.network,
            x_start=vertices,
            t=t,
            tongue_weight=self.cfg_train.TONGUE_WEIGHT,
            jaw_weight=self.cfg_train.JAW_WEIGHT,
            model_kwargs={
                "audio": audio,
                "actor_vec": actor_vec,
                "emotion_vec": emotion_vec,
            },
        )

        diffusion_loss, model_output = output["loss"], output["model_output"]
        if len(batch.actor_name) > 1:
            raise ValueError(f"Only batch size of 1 is supported. Current batch size: {len(batch.actor_name)}")
        # TODO for batch_size > 1 need to use the whole batch.actor_name
        additional_losses = self.compute_additional_losses(model_output, vertices, batch.actor_name[0])

        loss = torch.mean(diffusion_loss)
        loss += self.cfg_train.LOSS_LIP_DIST_ALPHA * additional_losses["lip_dist"]
        loss += self.cfg_train.LOSS_VELOCITY_ALPHA * additional_losses["velocity"]
        loss += self.cfg_train.LOSS_EXP_SMOOTH_ALPHA * additional_losses["expression_smooth"]

        loss.backward()
        if it % self.cfg_train.GRADIENT_ACCUMULATION_STEPS == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if it % self.cfg_train.LOG_PERIOD == 0:
            loss_value = loss.item()

            self.writer.add_scalar("loss", loss_value, self.global_step)

            if not utils.is_partial_exposure():
                self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
                self.writer.add_scalar("losses/diffusion", diffusion_loss.item(), self.global_step)
                if self.cfg_train.LOSS_LIP_DIST_ALPHA is not None and self.lip_dist_verts is not None:
                    self.writer.add_scalar("losses/lip_dist", additional_losses["lip_dist"].item(), self.global_step)
                if self.cfg_train.LOSS_VELOCITY_ALPHA is not None:
                    self.writer.add_scalar("losses/velocity", additional_losses["velocity"].item(), self.global_step)
                if self.cfg_train.LOSS_EXP_SMOOTH_ALPHA is not None and self.softmask_upperface is not None:
                    self.writer.add_scalar(
                        "losses/expression_smooth", additional_losses["expression_smooth"].item(), self.global_step
                    )

            time_remain = (time.time() - self.time_start) * (
                float(self.num_batches * self.cfg_train.NUM_EPOCHS) / float(self.global_step + 1) - 1.0
            )
            time_remain_str = str(datetime.timedelta(seconds=round(time_remain)))
            logging.info(
                "Epoch: {:3d}/{} | Iter: {:4d}/{} | Loss = {:8.5f} | Remaining time: {}".format(
                    epoch, self.cfg_train.NUM_EPOCHS, it, self.num_batches, loss_value, time_remain_str
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

        utils.validate_cfg_mod(self.cfg_train_mod, config_train_diffusion_base, "train")
        utils.validate_cfg_mod(self.cfg_dataset_mod, config_dataset_base, "dataset")
        utils.validate_cfg_mod(self.cfg_inference_mod, config_inference_diffusion_base, "inference")
        self.cfg_train = utils.module_to_easy_dict(config_train_diffusion_base, modifier=self.cfg_train_mod)
        self.cfg_dataset = utils.module_to_easy_dict(config_dataset_base, modifier=self.cfg_dataset_mod)
        self.cfg_inference = utils.module_to_easy_dict(config_inference_diffusion_base, modifier=self.cfg_inference_mod)
        for actor_name, preproc_run_name_full in self.cfg_train.PREPROC_RUN_NAME_FULL.items():
            if preproc_run_name_full.startswith("XXXXXX_XXXXXX"):
                raise ValueError(f'Please update PREPROC_RUN_NAME_FULL in config_train.py for actor "{actor_name}"')
            utils.validate_identifier_or_raise(preproc_run_name_full, f'PREPROC_RUN_NAME_FULL for actor "{actor_name}"')

        self.setup_torch()

        self.audio_provider = AudioProvider(self.cfg_train, self.cfg_dataset)
        self.target_provider = TargetProviderFullFace(self.cfg_train, self.cfg_dataset)
        self.dataset = AnimationSegmentDataset(
            self.cfg_train,
            self.cfg_dataset,
            self.audio_provider,
            None,
            self.target_provider,
        )
        self.dataset_loader = torch.utils.data.DataLoader(self.dataset, self.cfg_train.BATCH_SIZE, shuffle=True)
        self.num_batches = len(self.dataset_loader)

        network_info_full, network_info_deploy = self.compose_network_info()
        self.export_meta_data(network_info_full, network_info_deploy)
        self.network_params = network_info_full["params"]
        self.network_extra_data = self.compose_network_extra_data()

        self.network = self.create_network()
        self.network = self.network.cuda()

        if self.cfg_train.PRETRAINED_NET_FPATH is not None:
            logging.info(f"Using pre-trained weights: {self.cfg_train.PRETRAINED_NET_FPATH}")
            self.network.load_state_dict(torch.load(self.cfg_train.PRETRAINED_NET_FPATH))
        else:
            logging.info("Training the network from scratch")

        self.diffusion = create_gaussian_diffusion(
            steps=self.cfg_train.DIFFUSION_STEPS,
            noise_schedule=self.cfg_train.DIFFUSION_NOISE_SCHEDULE,
            predict_xstart=True,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            timestep_respacing=self.cfg_train.TIMESTEP_RESPACING,
        )

        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)

        self.optimizer = optim.Adam(
            filter(lambda param: param.requires_grad, self.network.parameters()), lr=self.cfg_train.LEARNING_RATE
        )
        self.lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.cfg_train.NUM_EPOCHS,
            lr_min=1e-5,
            warmup_lr_init=1e-4,
            warmup_t=10,
            cycle_limit=1,
            t_in_epochs=True,
        )

        # TODO Implement support for per-actor meta info usage
        actor_name_0 = self.cfg_dataset.ACTOR_NAMES[0]
        if len(self.cfg_dataset.ACTOR_NAMES) > 1:
            logging.warning(f"Using meta information from actor [0] ({actor_name_0}) for all actors")

        if self.cfg_train.LOSS_LIP_DIST_ALPHA is not None:
            self.lip_dist_verts = self.read_lip_dist_verts(actor_name_0)  # Returns None if not provided

        if self.cfg_train.LOSS_EXP_SMOOTH_ALPHA is not None:
            mean_face = self.target_provider.get_skin_pca_shapes(actor_name_0)["shapes_mean"]
            eye_dist_verts = self.read_eye_dist_verts(actor_name_0)
            self.softmask_upperface = None
            if eye_dist_verts is not None:
                y_thres, z_thres, sharpness = losses.cal_mask_params(mean_face, eye_dist_verts)
                self.softmask_upperface = losses.generate_vertex_weights_vectorized(
                    mean_face, y_thres, z_thres, sharpness
                )
                self.softmask_upperface = self.softmask_upperface.reshape(-1, 1)

        self.writer = SummaryWriter(self.out_dir)
        logging.info(f"Output directory: {self.out_dir}")
        logging.info("--------------------------------------------------------------------------------")

        self.network.train()

        self.global_step = 0

    def preload_dataset(self) -> None:
        self.dataset.preload()

    def run(self) -> dict:
        self.time_start = time.time()

        try:
            for epoch in range(self.cfg_train.NUM_EPOCHS):
                self.optimizer.zero_grad()
                for it, batch in enumerate(self.dataset_loader):
                    self.train_step(epoch, it, batch)
                self.lr_scheduler.step(epoch)
                # TODO add checkpoint save every K epochs
        except KeyboardInterrupt:
            logging.info("Training interrupted")

        training_time = time.time() - self.time_start
        training_time_str = str(datetime.timedelta(seconds=round(training_time)))
        logging.info("--------------------------------------------------------------------------------")
        logging.info(f"Total training time: {training_time_str} ({training_time:.2f} sec)")

        if not utils.is_partial_exposure():
            torch.save(self.network.state_dict(), os.path.join(self.out_dir, "weights.pth"))
        self.export_onnx()

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
    cfg_inference_mod: dict | None = None,
) -> dict:
    trainer = Trainer(cfg_train_mod, cfg_dataset_mod, cfg_inference_mod)
    trainer.setup()
    trainer.preload_dataset()
    result = trainer.run()
    return result
