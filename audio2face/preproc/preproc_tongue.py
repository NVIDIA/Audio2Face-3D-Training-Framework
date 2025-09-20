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
import time
import logging
import json
import numpy as np
from scipy import spatial

from audio2face import utils
from audio2face.config_base import config_preproc_base, config_dataset_base
from audio2face.geometry import anim_cache, maya_cache, pca
from audio2face.geometry.xform import rigidXform


def decoupleTransformDeformation(
    tongue_cache_dir: str,
    target_seqs: list[str],
    tongue_neutral_fpath: str,
    out_tongue_cache_dir: str,
    rigid_vertex_list_fpath: str,
    fps: float,
    out_trans_delfa_fpath: str | None = None,
) -> None:
    logging.info("Running decoupling transform deformation...")

    if not os.path.exists(out_tongue_cache_dir):
        os.mkdir(out_tongue_cache_dir)

    neutral_tongue_vtxMat = np.load(tongue_neutral_fpath)

    with open(rigid_vertex_list_fpath) as f:
        rigid_vertex_list = json.load(f)
    rigid_neutral_tongue = neutral_tongue_vtxMat[rigid_vertex_list, :]

    deltaDict = {}

    for ci in range(len(target_seqs)):
        cache_fpath = utils.get_cache_fpath(tongue_cache_dir, target_seqs[ci])
        data = anim_cache.read_cache(cache_fpath)
        logging.info(
            f'[{ci}/{len(target_seqs)}] Processing shot "{target_seqs[ci]}" with shape {data.shape} at {cache_fpath}'
        )

        rigid_data = data[:, rigid_vertex_list, :]

        transDelta = rigid_data - rigid_neutral_tongue.reshape(
            1, rigid_neutral_tongue.shape[0], rigid_neutral_tongue.shape[1]
        )
        deltaDict[target_seqs[ci]] = transDelta

        stab_tongue_data = []
        for fr in range(rigid_data.shape[0]):
            RR, tt = rigidXform(rigid_neutral_tongue, rigid_data[fr, ...])
            alignTongue = np.dot(RR.T, data[fr, ...].T).T + tt
            stab_tongue_data.append(alignTongue)

        outCacheFilePath = os.path.join(out_tongue_cache_dir, "{}/{}.xml".format(target_seqs[ci], target_seqs[ci]))
        if config_preproc_base.VERBOSE:
            logging.info(f"Saving the stabilized cache to {outCacheFilePath}")
        if not os.path.exists(os.path.dirname(outCacheFilePath)):
            os.mkdir(os.path.dirname(outCacheFilePath))

        maya_cache.export_animation(stab_tongue_data, fps, outCacheFilePath)

    if out_trans_delfa_fpath is not None:
        # this saves out the inverse translation of the tongue back to the neutral position
        np.savez(out_trans_delfa_fpath, vtx_list=rigid_vertex_list, neutral=rigid_neutral_tongue, **deltaDict)


def computePcaVectors(
    tongue_cache_dir,
    target_seqs,
    tongue_neutral_fpath,
    out_pca_inter_fpath,
    out_pca_fpath,
    variance_threshold,
    force_components=None,
):
    logging.info("Running pruning...")

    verts_all = []

    tongue_neutral = np.load(tongue_neutral_fpath).reshape(1, -1)
    prune_dist = 10.0

    for ci in range(len(target_seqs)):
        cache_fpath = utils.get_cache_fpath(tongue_cache_dir, target_seqs[ci])
        data = anim_cache.read_cache(cache_fpath)
        logging.info(
            f'[{ci}/{len(target_seqs)}] Processing shot "{target_seqs[ci]}" with shape {data.shape} at {cache_fpath}'
        )

        # sample every 1 frames
        data = data[::1, ...]
        if config_preproc_base.VERBOSE:
            logging.info(f"Shape after time down-sampling is {data.shape}")

        data2d = data.reshape(data.shape[0], -1)

        numFr = data2d.shape[0]
        tree = spatial.cKDTree(data2d)
        closeIdxList = []
        for i in range(0, numFr, 10):
            idxList = tree.query_ball_point(data2d[i, :], r=prune_dist)
            closeIdxList += idxList
        closeIdxList = list(set(closeIdxList))

        validList_local = list(range(0, numFr, 10))
        for i in range(numFr):
            if i not in closeIdxList:
                validList_local.append(i)
        validList_local = sorted(validList_local)

        data = data[validList_local, ...]
        if config_preproc_base.VERBOSE:
            logging.info(f"Shape after pruning is {data.shape}")

        verts_all.append(data.copy())

    verts_all = np.concatenate(verts_all, axis=0)
    if config_preproc_base.VERBOSE:
        logging.info(f"The shape of all pruned poses stacked is {verts_all.shape}")

    # =====================================================

    verts_all = verts_all.reshape(verts_all.shape[0], -1)

    logging.info("Running PCA decomposition...")
    logging.info(f"Computing PCA for a matrix with shape {verts_all.shape}...")
    if True:  # TODO make a flag in the config
        evecs_tongue, evals_tongue, mean_tongue = pca.pca_truncated(
            verts_all,
            variance_threshold=variance_threshold,
            custom_mean=tongue_neutral,
            force_components=force_components,
            use_cupy=False,
        )
    else:
        evecs_tongue, evals_tongue, mean_tongue = pca.pca_truncated(
            verts_all,
            variance_threshold=variance_threshold,
            custom_mean=None,
            force_components=force_components,
            use_cupy=False,
        )
    logging.info(
        f'Truncated PCA Decomp: Based on [var = {variance_threshold}], # of components = {evecs_tongue.shape[1]}{" (forced)" if force_components is not None else ""}'
    )

    logging.info(f"Result PCA dimensions: Eigen vectors: {evecs_tongue.shape}")
    logging.info(f"Result PCA dimensions: Eigen values: {evals_tongue.shape}")
    logging.info(f"Result PCA dimensions: Mean: {mean_tongue.shape}")
    evecs_tongue = utils.convert_to_float32(evecs_tongue)
    evals_tongue = utils.convert_to_float32(evals_tongue)
    mean_tongue = utils.convert_to_float32(mean_tongue)
    logging.info(f"Converted data to float32")
    np.savez(out_pca_inter_fpath, evecs=evecs_tongue, evals=evals_tongue, mean=mean_tongue)

    shapes_matrix = evecs_tongue.T.reshape(evecs_tongue.T.shape[0], -1, 3)
    shapes_mean = mean_tongue.reshape(-1, 3)
    np.savez(out_pca_fpath, shapes_matrix=shapes_matrix, shapes_mean=shapes_mean)


def exportReconTonguePca(
    tongue_cache_dir: str,
    target_seqs: list[str],
    out_pca_inter_fpath: str,
    out_pca_coeff_fpath: str,
    num_dims: int | None = None,
) -> None:
    logging.info("Computing PCA coeffs...")

    tonguePcaData = np.load(out_pca_inter_fpath)
    if num_dims is None:
        num_dims = tonguePcaData["evecs"].shape[1]
    evecs_tongue = tonguePcaData["evecs"][:, :num_dims]
    mean_tongue = tonguePcaData["mean"]

    pcaCoeffDict = {}

    for ci in range(len(target_seqs)):
        cache_fpath = utils.get_cache_fpath(tongue_cache_dir, target_seqs[ci])
        data = anim_cache.read_cache(cache_fpath)
        data = data.reshape(data.shape[0], -1)
        logging.info(
            f'[{ci}/{len(target_seqs)}] Processing shot "{target_seqs[ci]}" with shape {data.shape} at {cache_fpath}'
        )

        delta_mean = data - mean_tongue
        coeff = np.dot(evecs_tongue.T, delta_mean.T).T
        if config_preproc_base.VERBOSE:
            logging.info(f"Coeffs shape: {coeff.shape}")
        pcaCoeffDict[target_seqs[ci]] = coeff

    np.savez(out_pca_coeff_fpath, **pcaCoeffDict)


def run(
    actor_name: str,
    run_name_full: str,
    cfg_preproc_mod: dict | None = None,
    cfg_dataset_mod: dict | None = None,
) -> dict:
    cfg_preproc = utils.module_to_easy_dict(config_preproc_base, modifier=cfg_preproc_mod)
    cfg_dataset = utils.module_to_easy_dict(config_dataset_base, modifier=cfg_dataset_mod)

    utils.validate_identifier_or_raise(run_name_full, "Preproc Run Name Full")

    out_dir = os.path.normpath(os.path.join(cfg_preproc.PREPROC_OUTPUT_ROOT, run_name_full, "pca", "tongue"))
    inter_dir = os.path.normpath(os.path.join(out_dir, "intermediate"))
    tongue_pca_shapes_fpath = os.path.join(out_dir, "tongue_pca_shapes.npz")
    tongue_pca_coeffs_fpath = os.path.join(out_dir, "tongue_pca_coeffs_all.npz")
    tongue_cache_stab_root = os.path.join(inter_dir, "tongue_cache_stab")
    tongue_pca_shapes_inter_fpath = os.path.join(inter_dir, "tongue_pca_shapes_inter.npz")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    logging.info("--------------------------------------------------------------------------------")
    logging.info("Tongue data preprocessing run: {}".format(run_name_full))
    logging.info("--------------------------------------------------------------------------------")

    # Check if tongue data is available
    tongue_cache_root = cfg_dataset.TONGUE_CACHE_ROOT.get(actor_name)
    tongue_neutral_fpath = cfg_dataset.TONGUE_NEUTRAL_FPATH.get(actor_name)
    tongue_rigid_vertex_list_fpath = cfg_dataset.TONGUE_RIGID_VERTEX_LIST_FPATH.get(actor_name)

    # Check if all required tongue data is available
    missing_data = (
        tongue_cache_root is None
        or tongue_neutral_fpath is None
        or tongue_rigid_vertex_list_fpath is None
        or not os.path.exists(tongue_cache_root)
        or not os.path.exists(tongue_neutral_fpath)
        or not os.path.exists(tongue_rigid_vertex_list_fpath)
    )

    if missing_data:
        logging.info("Tongue data not available for actor '{}', skipping tongue preprocessing".format(actor_name))
        logging.info("--------------------------------------------------------------------------------")
        logging.info("Tongue preprocessing skipped - synthetic data generated")
        logging.info("--------------------------------------------------------------------------------")
        np.savez(
            tongue_pca_shapes_fpath,
            shapes_matrix=np.zeros(cfg_preproc.DEFAULT_TONGUE_PCA_SHAPE, dtype=np.float32),
            shapes_mean=np.zeros(cfg_preproc.DEFAULT_TONGUE_PCA_SHAPE[1:], dtype=np.float32),
        )
        np.savez(
            tongue_pca_coeffs_fpath,
            placeholder=np.zeros((1, cfg_preproc.DEFAULT_TONGUE_PCA_SHAPE[0]), dtype=np.float32),
        )

        return {
            "processing_time": 0.0,
            "out_dir": out_dir,
            "inter_dir": inter_dir,
            "skipped": True,
        }

    time_start = time.time()

    if cfg_preproc.TONGUE_CACHE_SHOTS.get(actor_name) is None:
        tongue_cache_shots = utils.get_all_subdir_names(tongue_cache_root)
        if len(tongue_cache_shots) == 0:
            raise RuntimeError(f"Unable to find any tongue caches at the directory: {tongue_cache_root}")
    else:
        tongue_cache_shots = cfg_preproc.TONGUE_CACHE_SHOTS.get(actor_name)

    # # 1. decouple transform component from the tongue (derived from lower denture)
    decoupleTransformDeformation(
        tongue_cache_root,
        tongue_cache_shots,
        tongue_neutral_fpath,
        tongue_cache_stab_root,
        tongue_rigid_vertex_list_fpath,
        cfg_dataset.CACHE_FPS[actor_name],
    )

    # 2. compute
    computePcaVectors(
        tongue_cache_stab_root,
        tongue_cache_shots,
        tongue_neutral_fpath,
        tongue_pca_shapes_inter_fpath,
        tongue_pca_shapes_fpath,
        variance_threshold=cfg_preproc.TONGUE_PCA_VARIANCE_THRESHOLD,
        force_components=cfg_preproc.TONGUE_FORCE_COMPONENTS.get(actor_name),
    )

    # # 3. export pca coeffs
    exportReconTonguePca(
        tongue_cache_stab_root,
        tongue_cache_shots,
        tongue_pca_shapes_inter_fpath,
        tongue_pca_coeffs_fpath,
        num_dims=None,
    )

    processing_time = time.time() - time_start

    logging.info("--------------------------------------------------------------------------------")
    logging.info("Total processing time: {:.2f} sec".format(processing_time))
    logging.info("Data artifacts were saved to {}".format(out_dir))
    logging.info("--------------------------------------------------------------------------------")

    return {
        "processing_time": processing_time,
        "out_dir": out_dir,
        "inter_dir": inter_dir,
        "skipped": False,
    }
