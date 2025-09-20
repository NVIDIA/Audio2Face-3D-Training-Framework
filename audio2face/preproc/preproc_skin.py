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
import glob
import time
import json
import logging
import numpy as np
import cupy as cp
from scipy import spatial

from audio2face import utils
from audio2face.config_base import config_preproc_base, config_dataset_base
from audio2face.geometry import pca, anim_cache


def get_any_cache_info(cache_root: str) -> dict:
    found_items = glob.glob(os.path.join(cache_root, "*"))
    if len(found_items) == 0:
        raise RuntimeError(f"Unable to find any shot cache at the directory: {cache_root}")
    shot_dir = found_items[0]
    shot_name = os.path.split(shot_dir)[1]
    cache_fpath = utils.get_cache_fpath(cache_root, shot_name)
    data = anim_cache.read_cache(cache_fpath)
    return {"num_verts": data.shape[1]}


def get_mesh_mask(num_verts: int, mesh_mask_fpath: str | None = None) -> np.ndarray:
    if mesh_mask_fpath is None:
        mesh_mask = np.ones(num_verts * 3, dtype=bool)
    else:
        mesh_mask_vtx_idx = np.load(mesh_mask_fpath)
        logging.info(f"Using mesh mask: {mesh_mask_fpath}")
        mesh_mask = np.zeros(num_verts * 3, dtype=bool)
        for idx in mesh_mask_vtx_idx:
            mesh_mask[idx * 3 : (idx + 1) * 3] = True
    return mesh_mask


def pruneSimilarPoses(
    cache_root: str,
    target_seqs: list[str],
    prune_mesh_mask: np.ndarray,
    out_pose_idx_json_fpath: str,
    prune_dist: float = 2.0,
) -> None:
    """
    prune out similar pose frames in a sequence using KD_tree and L2 distance
    """

    logging.info("Pruning similar poses...")

    total_frame_num = 0
    initial_total_frame_num = 0
    validIdxAll = []
    for si, seq in enumerate(target_seqs):
        cache_fpath = utils.get_cache_fpath(cache_root, seq)
        data = anim_cache.read_cache(cache_fpath)
        initial_total_frame_num += data.shape[0]
        logging.info(f'[{si}/{len(target_seqs)}] Processing shot "{seq}" with shape {data.shape} at {cache_fpath}')
        poseMat = data.reshape(data.shape[0], -1)[:, prune_mesh_mask]
        poseMatOrg = poseMat.copy()

        numSelPoses = poseMat.shape[0]
        prevNumSelPoses = 0
        validList = [i for i in range(poseMatOrg.shape[0])]
        while numSelPoses != prevNumSelPoses and len(validList) > 10:
            prevNumSelPoses = numSelPoses
            poseMat = poseMatOrg[validList, :]

            tree = spatial.cKDTree(poseMat)
            numFr = poseMat.shape[0]

            closeIdxList = []
            for i in range(0, numFr, 10):
                idxList = tree.query_ball_point(poseMat[i, :], r=prune_dist)
                closeIdxList += idxList

            closeIdxList = list(set(closeIdxList))

            validList_local = list(range(0, numFr, 10))
            for i in range(numFr):
                if i not in closeIdxList:
                    validList_local.append(i)
            validList_local = sorted(validList_local)

            validList = [validList[vi] for vi in validList_local]
            numSelPoses = len(validList)

        for vi in validList:
            validIdxAll.append([si, vi])

        if config_preproc_base.VERBOSE:
            logging.info(f"Output number of poses: {len(validList)}")

        total_frame_num += len(validList)

    logging.info(f"Initial total number of frames: {initial_total_frame_num}")
    logging.info(f"Result total number of frames: {total_frame_num}")

    re_outPoseIdxDict = {}
    for si, seq in enumerate(target_seqs):
        re_outPoseIdxDict[seq] = []

    for si, vIdx in validIdxAll:
        seqName = target_seqs[si]
        re_outPoseIdxDict[seqName].append(vIdx)

    with open(out_pose_idx_json_fpath, "w") as outfile:
        json.dump(re_outPoseIdxDict, outfile, sort_keys=True, indent=4)


def selectDistinctPosePca(
    cache_root,
    target_seqs,
    in_pose_idx_json_fpath,
    prune_mesh_mask,
    pca_out_pose_mat_fpath,
    pose_mat_all_fpath=None,
    max_iter=500,
):
    """
    read input .npy files and output selected index as dictionary
    """

    logging.info("Selecting distinct poses with PCA...")

    # concatenate all poses
    with open(in_pose_idx_json_fpath) as f:
        selPoseIdxDict = json.load(f)

    validIdxBnd = [0]  # per-pose valid index numbers
    numVtx3 = np.sum(prune_mesh_mask)  # active mask
    poseMatAll = np.zeros((0, numVtx3), dtype=np.float32)

    for si, seq in enumerate(target_seqs):
        validIdx = selPoseIdxDict[seq]
        validIdxBnd.append(len(validIdx))

        if len(validIdx) == 0:
            continue

        if pose_mat_all_fpath is None or not os.path.exists(pose_mat_all_fpath):
            cache_fpath = utils.get_cache_fpath(cache_root, seq)
            data = anim_cache.read_cache(cache_fpath)
            logging.info(f'[{si}/{len(target_seqs)}] Processing shot "{seq}" with shape {data.shape} at {cache_fpath}')
            poseMat = data.reshape(data.shape[0], -1)[:, prune_mesh_mask]
            poseMatAll = np.vstack((poseMatAll, poseMat[validIdx, :]))
            if config_preproc_base.VERBOSE:
                logging.info(f"The shape of all pruned poses stacked is {poseMatAll.shape}")

    validIdxBnd = np.cumsum(validIdxBnd)

    if pose_mat_all_fpath is None or not os.path.exists(pose_mat_all_fpath):
        np.save(pose_mat_all_fpath, poseMatAll)
    else:
        poseMatAll = np.load(pose_mat_all_fpath)

    # test using pca
    numFr = poseMatAll.shape[0]
    poseMatAll_cp = cp.array(poseMatAll)

    variance_threshold = 0.9999
    numInitFr = 4
    idxMult = numFr / numInitFr
    initIdx = [int(idxMult * i) for i in range(numInitFr)]

    autoSelIdx = []

    for ii in range(max_iter):
        # init compression
        mat = poseMatAll_cp[np.array(initIdx + autoSelIdx, dtype=int), :]
        evecs_t, evals_t, mean = pca.pca_truncated(
            mat,
            variance_threshold=variance_threshold,
            custom_mean=None,
            force_components=None,
            use_cupy=True,
        )

        # check reconError
        deltaPoseMat = poseMatAll_cp - mean[np.newaxis, :]
        deltaPoseMat_T = deltaPoseMat.T
        coeff = cp.dot(evecs_t.T, deltaPoseMat_T)
        diffMat = cp.dot(evecs_t, coeff) - deltaPoseMat_T
        errVec = cp.linalg.norm(diffMat, axis=0)
        errVec[np.array(initIdx + autoSelIdx, dtype=int)] = 0.0

        autoSelIdx.append(np.argmax(cp.asnumpy(errVec)))
        logging.info(
            f"Iteration [{ii}/{max_iter}] | Matrix {mat.shape} -> {evecs_t.shape[1]} components | max error = {cp.max(errVec)}"
        )

        if ii % 100 == 99:
            re_outPoseIdxDict = {}
            for si, seq in enumerate(target_seqs):
                re_outPoseIdxDict[seq] = []

            for sIdx in initIdx + autoSelIdx:
                fIdx = np.min(np.where(validIdxBnd > sIdx)[0]) - 1
                idx_inFile = sIdx - validIdxBnd[fIdx]
                seq = target_seqs[fIdx]
                validIdx = selPoseIdxDict[seq]
                re_outPoseIdxDict[seq].append(validIdx[idx_inFile])

            if config_preproc_base.VERBOSE:
                with open(pca_out_pose_mat_fpath.format(ii), "w") as outfile:
                    json.dump(re_outPoseIdxDict, outfile, sort_keys=True, indent=4)

    re_outPoseIdxDict = {}
    for si, seq in enumerate(target_seqs):
        re_outPoseIdxDict[seq] = []

    for sIdx in initIdx + autoSelIdx:
        fIdx = np.min(np.where(validIdxBnd > sIdx)[0]) - 1
        idx_inFile = sIdx - validIdxBnd[fIdx]
        seq = target_seqs[fIdx]
        validIdx = selPoseIdxDict[seq]
        re_outPoseIdxDict[seq].append(validIdx[idx_inFile])

    for key in re_outPoseIdxDict.keys():
        re_outPoseIdxDict[key] = sorted(re_outPoseIdxDict[key])

    with open(pca_out_pose_mat_fpath.format("final"), "w") as outfile:
        json.dump(re_outPoseIdxDict, outfile, sort_keys=True, indent=4)


def runPca(
    cache_root,
    target_seqs,
    pca_prune_pose_idx_json_fpath,
    skin_neutral_fpath,
    out_pca_data_inter_path,
    out_pca_data_path,
    variance_threshold,
    force_components=None,
):
    logging.info("Running PCA decomposition...")
    # read neutral pose, which will be the user mean - we obtained it from one of the 4d capture sequence.
    if skin_neutral_fpath is not None:
        neutral_pose = np.load(skin_neutral_fpath)
        custom_mean = neutral_pose.flatten()
    else:
        custom_mean = None

    # concatenate all poses
    with open(pca_prune_pose_idx_json_fpath.format("final")) as f:
        selPoseIdxDict = json.load(f)

    poseMatAll = []

    for si, seq in enumerate(target_seqs):
        validIdx = selPoseIdxDict[seq]
        cache_fpath = utils.get_cache_fpath(cache_root, seq)
        data = anim_cache.read_cache(cache_fpath)
        logging.info(f'[{si}/{len(target_seqs)}] Processing shot "{seq}" with shape {data.shape} at {cache_fpath}')

        # check the validIdx with the data length
        invalidIdx = np.where(np.array(validIdx) >= data.shape[0])[0]
        if len(invalidIdx) > 0:
            for idx in invalidIdx:
                validIdx.remove(validIdx[idx])
                logging.info(f"Removed invalid index: {idx}/{data.shape[0]}")

        poseMat = data.reshape(data.shape[0], -1)[validIdx, :]
        if config_preproc_base.VERBOSE:
            logging.info(f"The shape of pruned matrix for the shot is {poseMat.shape}")
        poseMatAll.append(poseMat.astype(np.float32))

    poseMatAll = np.concatenate(poseMatAll, axis=0)
    if config_preproc_base.VERBOSE:
        logging.info(f"The shape of all pruned poses stacked is {poseMatAll.shape}")

    logging.info(f"Computing PCA for a matrix with shape {poseMatAll.shape}...")
    evecs_t, evals_t, mean = pca.pca_truncated(
        poseMatAll,
        variance_threshold=variance_threshold,
        custom_mean=custom_mean,
        force_components=force_components,
        use_cupy=False,
    )
    logging.info(
        f'Truncated PCA Decomp: Based on [var = {variance_threshold}], # of components = {evecs_t.shape[1]}{" (forced)" if force_components is not None else ""}'
    )

    logging.info(f"Result PCA dimensions: Eigen vectors: {evecs_t.shape}")
    logging.info(f"Result PCA dimensions: Eigen values: {evals_t.shape}")
    logging.info(f"Result PCA dimensions: Mean: {mean.shape}")

    evecs_t = utils.convert_to_float32(evecs_t)
    evals_t = utils.convert_to_float32(evals_t)
    mean = utils.convert_to_float32(mean)
    logging.info(f"Converted data to float32")
    np.savez(
        out_pca_data_inter_path, evecs_t=evecs_t, evals_t=evals_t, mean=mean, variance_threshold=variance_threshold
    )

    shapes_matrix = evecs_t.T.reshape(evecs_t.T.shape[0], -1, 3)
    shapes_mean = mean.reshape(-1, 3)
    np.savez(out_pca_data_path, shapes_matrix=shapes_matrix, shapes_mean=shapes_mean)


def computePcaCoeffs(
    cache_root, target_seqs, pca_data_inter_path, pca_coeff_path, num_basis=None, new_target_seqs=None
):
    logging.info("Computing PCA coeffs...")

    pca_data = np.load(pca_data_inter_path)
    if num_basis is None:
        evecs_t = pca_data["evecs_t"]
    else:
        evecs_t = pca_data["evecs_t"][:, :num_basis]
    mean = pca_data["mean"]

    coeffDict = {}
    for si, seq in enumerate(target_seqs):
        cache_fpath = utils.get_cache_fpath(cache_root, seq)
        data = anim_cache.read_cache(cache_fpath)
        logging.info(f'[{si}/{len(target_seqs)}] Processing shot "{seq}" with shape {data.shape} at {cache_fpath}')
        data_flat = data.reshape(data.shape[0], -1)

        # pca projection
        pcaCoeffs = np.dot(evecs_t.T, (data_flat - mean[np.newaxis, :]).T).T
        if config_preproc_base.VERBOSE:
            logging.info(f"Coeffs shape: {pcaCoeffs.shape}")

        if new_target_seqs is not None:
            coeffDict[new_target_seqs[si]] = pcaCoeffs
        else:
            coeffDict[seq] = pcaCoeffs

    np.savez(pca_coeff_path, **coeffDict)


def run(
    actor_name: str,
    run_name_full: str,
    cfg_preproc_mod: dict | None = None,
    cfg_dataset_mod: dict | None = None,
) -> dict:
    cfg_preproc = utils.module_to_easy_dict(config_preproc_base, modifier=cfg_preproc_mod)
    cfg_dataset = utils.module_to_easy_dict(config_dataset_base, modifier=cfg_dataset_mod)

    utils.validate_identifier_or_raise(run_name_full, "Preproc Run Name Full")

    out_dir = os.path.normpath(os.path.join(cfg_preproc.PREPROC_OUTPUT_ROOT, run_name_full, "pca", "skin"))
    inter_dir = os.path.normpath(os.path.join(out_dir, "intermediate"))
    skin_pca_shapes_fpath = os.path.join(out_dir, "skin_pca_shapes.npz")
    skin_pca_coeffs_fpath = os.path.join(out_dir, "skin_pca_coeffs_all.npz")
    skin_pruned_poses_sim_fpath = os.path.join(inter_dir, "skin_pruned_poses_sim.json")
    skin_pruned_poses_pca_fpath = os.path.join(inter_dir, "skin_pruned_poses_pca_{}.json")
    skin_cache_all_fpath = os.path.join(inter_dir, "skin_cache_all.npy")
    skin_pca_shapes_inter_fpath = os.path.join(inter_dir, "skin_pca_shapes_inter.npz")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    logging.info("--------------------------------------------------------------------------------")
    logging.info("Skin data preprocessing run: {}".format(run_name_full))
    logging.info("--------------------------------------------------------------------------------")

    time_start = time.time()

    if cfg_preproc.SKIN_CACHE_SHOTS.get(actor_name) is None:
        skin_cache_shots = utils.get_all_subdir_names(cfg_dataset.SKIN_CACHE_ROOT[actor_name])
        if len(skin_cache_shots) == 0:
            raise RuntimeError(
                f"Unable to find any skin caches at the directory: {cfg_dataset.SKIN_CACHE_ROOT[actor_name]}"
            )
    else:
        skin_cache_shots = cfg_preproc.SKIN_CACHE_SHOTS.get(actor_name)

    if cfg_preproc.SKIN_PRUNE_CACHE_ROOT.get(actor_name) is None:
        skin_prune_cache_root = cfg_dataset.SKIN_CACHE_ROOT[actor_name]
    else:
        skin_prune_cache_root = cfg_preproc.SKIN_PRUNE_CACHE_ROOT.get(actor_name)

    if cfg_preproc.SKIN_PRUNE_CACHE_SHOTS.get(actor_name) is None:
        skin_prune_cache_shots = utils.get_all_subdir_names(skin_prune_cache_root)
    else:
        skin_prune_cache_shots = cfg_preproc.SKIN_PRUNE_CACHE_SHOTS.get(actor_name)

    skin_prune_mesh_num_verts = get_any_cache_info(skin_prune_cache_root)["num_verts"]
    if skin_prune_cache_root == cfg_preproc.SKIN_PRUNE_CACHE_ROOT.get(actor_name):
        logging.info(f"Using different resolution for pruning: {skin_prune_mesh_num_verts} vertices")

    skin_prune_mesh_mask = get_mesh_mask(
        skin_prune_mesh_num_verts, cfg_preproc.SKIN_PRUNE_MESH_MASK_FPATH.get(actor_name)
    )
    skin_prune_mesh_mask_num_verts = int(np.sum(skin_prune_mesh_mask) / 3)
    if not all(skin_prune_mesh_mask):
        logging.info(f"Using mesh mask: {skin_prune_mesh_num_verts} -> {skin_prune_mesh_mask_num_verts} vertices")

    # ----------------------------------------------------
    # # 1. prune similar pose using kd tree
    pruneSimilarPoses(
        skin_prune_cache_root,
        skin_prune_cache_shots,
        skin_prune_mesh_mask,
        skin_pruned_poses_sim_fpath,
        cfg_preproc.SKIN_PRUNE_SIM_DIST,
    )

    # # 2. select poses using PCA
    selectDistinctPosePca(
        skin_prune_cache_root,
        skin_prune_cache_shots,
        skin_pruned_poses_sim_fpath,
        skin_prune_mesh_mask,
        skin_pruned_poses_pca_fpath,
        pose_mat_all_fpath=skin_cache_all_fpath,
        max_iter=cfg_preproc.SKIN_SELECT_DISTINCT_MAX_ITER,
    )

    # 3. run PCA and build projection matrix
    runPca(
        cfg_dataset.SKIN_CACHE_ROOT[actor_name],
        skin_cache_shots,
        skin_pruned_poses_pca_fpath,
        cfg_dataset.SKIN_NEUTRAL_FPATH.get(actor_name),
        skin_pca_shapes_inter_fpath,
        skin_pca_shapes_fpath,
        variance_threshold=cfg_preproc.SKIN_PCA_VARIANCE_THRESHOLD,
        force_components=cfg_preproc.SKIN_FORCE_COMPONENTS.get(actor_name),
    )

    # # 4. get pca coefficients for network training
    computePcaCoeffs(
        cfg_dataset.SKIN_CACHE_ROOT[actor_name],
        skin_cache_shots,
        skin_pca_shapes_inter_fpath,
        skin_pca_coeffs_fpath,
        num_basis=None,
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
    }
