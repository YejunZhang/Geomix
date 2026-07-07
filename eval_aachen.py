#!/usr/bin/env python
"""
Aachen Day-Night visual localization pipeline for GeoMix.

The pipeline follows hloc (https://github.com/cvg/Hierarchical-Localization):
  1. Extract local features (SIFT / SuperPoint / R2D2 / DISK) for all images.
  2. Convert the reference NVM model to COLMAP format.
  3. Match database features and retriangulate the 3D model with the chosen
     detector, so the 2D and 3D sides use the same detector type.
  4. Match query keypoints to the retrieved 3D points with GeoMix and
     estimate poses with PnP + RANSAC.
Predicted query poses are written in the standard visuallocalization.net
submission format.

Requirements: hloc (with pycolmap), h5py, and this repository installed.

Usage:
    python eval_aachen.py --detector_2d sift --detector_3d sift
    python eval_aachen.py --detector_2d superpoint --detector_3d superpoint
"""

import argparse
import glob
import os
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from hloc import (
    extract_features,
    match_features,
    colmap_from_nvm,
    triangulation,
)

from geomix.utils.evaluator import load_matcher_from_ckpt, prepare_sample_inputs
from geomix.utils.geometry import estimate_pose

# Feature-extraction configurations per detector (hloc conventions)
FEATURE_CONFS = {
    "sift": {
        "output": "feats-sift",
        "model": {"name": "dog", "max_keypoints": 2048},
        "preprocessing": {"grayscale": True, "resize_max": 1024},
    },
    "superpoint": {
        "output": "feats-superpoint-n1024-r1024",
        "model": {"name": "superpoint", "nms_radius": 4, "max_keypoints": 2048, "keypoint_threshold": 0.005},
        "preprocessing": {"grayscale": True, "resize_max": 1024},
    },
    "r2d2": {
        "output": "feats-r2d2-n2048-r1024",
        "model": {"name": "r2d2", "max_keypoints": 2048},
        "preprocessing": {"grayscale": False, "resize_max": 1024},
    },
    "disk": {
        "output": "feats-disk-n2048-r1024",
        "model": {"name": "disk", "max_keypoints": 2048},
        "preprocessing": {"grayscale": False, "resize_max": 1024},
    },
}

SFM_DIR_NAMES = {
    "sift": "sfm_sift+NN-ratio",
    "superpoint": "sfm_superpoint+superglue",
    "r2d2": "sfm_r2d2+NN-mutual",
    "disk": "sfm_disk+NN-mutual",
}


def get_camera_matrix(camera_params):
    """Build the intrinsic matrix K from COLMAP camera parameters."""
    model = camera_params["model"]
    params = camera_params["params"]

    if model in ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]:
        f, cx, cy = params[0], params[1], params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif model in ["PINHOLE", "RADIAL"]:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        f, cx, cy = params[0], params[1], params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    return K


def normalize_keypoints(keypoints, K):
    """Convert pixel coordinates to normalized coordinates (bearing vectors)."""
    kp_homo = np.hstack([keypoints, np.ones((len(keypoints), 1))])
    bvecs = np.linalg.solve(K, kp_homo.T).T
    return bvecs[:, :2]


def project3d_to_bearing_vectors(pts3d, R, t):
    """Project 3D points into a camera frame and normalize to bearing vectors."""
    pts3d_cam = pts3d @ R.T + t
    pts3d_norm = pts3d_cam / pts3d_cam[:, -1, None]
    valid = pts3d_cam[:, -1] >= 0
    return pts3d_norm[:, :2], valid


def rotation_matrix_to_quaternion(R):
    """Convert a rotation matrix to a quaternion [qw, qx, qy, qz]."""
    from scipy.spatial.transform import Rotation

    quat = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def parse_query_file(query_file_path):
    """Parse an Aachen query file with image names and camera intrinsics."""
    queries = []
    with open(query_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            queries.append({
                "name": parts[0],
                "camera_model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": list(map(float, parts[4:])),
            })
    return queries


def parse_retrieval_pairs(pairs_file_path, topk):
    """Return the top-k retrieved database images for each query image."""
    retrieval_dict = defaultdict(list)
    with open(pairs_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            query_name, db_name = line.split()[:2]
            retrieval_dict[query_name].append(db_name)
    return {k: v[:topk] for k, v in retrieval_dict.items()}


def read_query_features(feature_file, query_name):
    """Read query keypoints and detector scores from an HDF5 feature file."""
    feat_group = feature_file[query_name]
    keypoints = feat_group["keypoints"][:]
    for score_key in ("scores", "keypoint_scores", "detection_scores"):
        if score_key in feat_group:
            return keypoints, feat_group[score_key][:], score_key
    return keypoints, np.ones(len(keypoints), dtype=np.float32), None


def main(args):
    dataset = Path(args.dataset_path)
    images = dataset / "images_upright/"

    detector_2d = args.detector_2d.lower()
    detector_3d = args.detector_3d.lower()

    outputs = Path(args.outputs) / f"{detector_2d}+{detector_3d}"
    outputs.mkdir(parents=True, exist_ok=True)

    reference_sfm = outputs / SFM_DIR_NAMES[detector_3d]
    sfm_pairs = Path(args.sfm_pairs)
    loc_pairs = Path(args.loc_pairs)
    for p in (sfm_pairs, loc_pairs):
        if not p.exists():
            raise FileNotFoundError(f"Missing pairs file: {p} (see hloc pairs/aachen)")

    feature_conf_3d = FEATURE_CONFS[detector_3d]
    features_3d = outputs / f"{feature_conf_3d['output']}.h5"
    feature_conf_2d = FEATURE_CONFS[detector_2d]
    features_2d = outputs / f"{feature_conf_2d['output']}.h5"

    matcher_conf = match_features.confs["NN-ratio"]

    print("=" * 60)
    print("Aachen Day-Night Pipeline Configuration")
    print("=" * 60)
    print(f"Dataset path: {dataset}")
    print(f"Output path: {outputs}")
    print(f"2D detector: {detector_2d}")
    print(f"3D detector: {detector_3d}")
    print("=" * 60)

    # ============ Step 1: Extract Local Features ============
    if features_3d.exists():
        print(f"\n[Step 1] 3D features exist, skipping: {features_3d}")
    else:
        print("\n[Step 1] Extracting 3D features...")
        features_3d = extract_features.main(feature_conf_3d, images, outputs)

    if detector_2d != detector_3d:
        if features_2d.exists():
            print(f"\n[Step 1b] 2D features exist, skipping: {features_2d}")
        else:
            print("\n[Step 1b] Extracting 2D features...")
            features_2d = extract_features.main(feature_conf_2d, images, outputs)
    else:
        features_2d = features_3d

    # ============ Step 2: Prepare reference SIFT model ============
    sfm_sift_model = outputs / "sfm_sift"
    if sfm_sift_model.exists():
        print("\n[Step 2] SfM SIFT model exists, skipping")
    else:
        print("\n[Step 2] Converting NVM to COLMAP...")
        colmap_from_nvm.main(
            dataset / "3D-models/aachen_cvpr2018_db.nvm",
            dataset / "3D-models/database_intrinsics.txt",
            dataset / "aachen.db",
            sfm_sift_model,
        )

    # ============ Step 3: Match Features for SfM ============
    sfm_matches_expected = outputs / f"{feature_conf_3d['output']}_{matcher_conf['output']}_{sfm_pairs.stem}.h5"
    if sfm_matches_expected.exists():
        print("\n[Step 3] SfM matches exist, skipping")
        sfm_matches = sfm_matches_expected
    else:
        print("\n[Step 3] Matching features...")
        sfm_matches = match_features.main(matcher_conf, sfm_pairs, feature_conf_3d["output"], outputs)

    # ============ Step 4: Retriangulate the 3D model ============
    reconstruction_files = [reference_sfm / "cameras.bin", reference_sfm / "images.bin", reference_sfm / "points3D.bin"]
    if all(f.exists() for f in reconstruction_files):
        print("\n[Step 4] Reconstruction exists, skipping")
    else:
        print("\n[Step 4] Triangulating 3D model...")
        triangulation.main(reference_sfm, sfm_sift_model, images, sfm_pairs, features_3d, sfm_matches)

    # ============ Load SfM model ============
    print("\n[Loading] Loading SfM model and features...")
    import pycolmap
    reconstruction = pycolmap.Reconstruction(reference_sfm)
    print(f"Loaded: {reference_sfm}")

    # ============ Step 5: GeoMix 2D-3D Matching Localization ============
    print("\n" + "=" * 60)
    print("[Step 5] Running GeoMix 2D-3D Matching Localization...")
    print("=" * 60)

    results_path = outputs / "Aachen_geomix.txt"

    retrieval_dict = parse_retrieval_pairs(loc_pairs, topk=args.topk)
    print(f"Loaded retrieval pairs for {len(retrieval_dict)} query images (top-{args.topk})")

    name_to_image = {img.name: img for img in reconstruction.images.values()}
    points3D_global = {pid: p3d.xyz for pid, p3d in reconstruction.points3D.items()}
    points3D_color_global = {pid: p3d.color for pid, p3d in reconstruction.points3D.items()}

    query_files = glob.glob(str(dataset / "queries/*_time_queries_with_intrinsics.txt"))
    all_queries = []
    for qf in query_files:
        all_queries.extend(parse_query_file(qf))
    print(f"Total {len(all_queries)} query images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher, _ = load_matcher_from_ckpt(args.weights)
    matcher = matcher.to(device).eval()
    print(f"Loaded weights: {args.weights}")

    sc_thres = 0.5
    results_list = []
    matching_times = []

    with h5py.File(features_2d, "r") as feature_file, torch.no_grad():
        for idx in tqdm(range(len(all_queries))):
            query = all_queries[idx]
            query_name = query["name"]

            if query_name not in retrieval_dict or query_name not in feature_file:
                continue

            keypoints_pix, scores_2d, _ = read_query_features(feature_file, query_name)

            # 2D keypoint colors
            img = Image.open(images / query_name).convert("RGB")
            img_arr = np.array(img)
            kp_x = np.clip(keypoints_pix[:, 0].astype(int), 0, img_arr.shape[1] - 1)
            kp_y = np.clip(keypoints_pix[:, 1].astype(int), 0, img_arr.shape[0] - 1)
            color2d_raw = img_arr[kp_y, kp_x].astype(np.float32)

            # Keep the top-scoring keypoints
            max_pts2d = 1024
            if len(keypoints_pix) > max_pts2d:
                top_indices = np.argsort(scores_2d)[::-1][:max_pts2d]
                keypoints_pix = keypoints_pix[top_indices]
                color2d_raw = color2d_raw[top_indices]

            K = get_camera_matrix({
                "model": query["camera_model"],
                "width": query["width"],
                "height": query["height"],
                "params": query["params"],
            })
            pts2d = normalize_keypoints(keypoints_pix, K).astype(np.float32)
            color2d = ((color2d_raw / 255.0) * 2 - 1).astype(np.float32)

            # Collect 3D points from the retrieved database images
            pts3d_global_dict = {}
            global_idx = 0
            covis_data = []

            for rank, db_name in enumerate(retrieval_dict[query_name]):
                if db_name not in name_to_image:
                    continue

                db_image = name_to_image[db_name]
                valid_p3d_ids = [p2d.point3D_id for p2d in db_image.points2D if p2d.has_point3D()]
                if len(valid_p3d_ids) < 10:
                    continue

                db_pts3d = np.array([points3D_global[pid] for pid in valid_p3d_ids], dtype=np.float32)
                db_pts3d_color = np.array([points3D_color_global[pid] for pid in valid_p3d_ids], dtype=np.float32)

                # Bearing vectors in the database-image frame
                cam_from_world = db_image.cam_from_world()
                db_R = cam_from_world.rotation.matrix()
                db_t = cam_from_world.translation
                db_pts3dm, valid_mask = project3d_to_bearing_vectors(db_pts3d, db_R, db_t)

                valid_p3d_ids = [pid for pid, v in zip(valid_p3d_ids, valid_mask) if v]
                db_pts3d = db_pts3d[valid_mask]
                db_pts3dm = db_pts3dm[valid_mask].astype(np.float32)
                db_pts3d_color = db_pts3d_color[valid_mask]
                if len(valid_p3d_ids) < 10:
                    continue

                db_pts3d_color_norm = ((db_pts3d_color / 255.0) * 2 - 1).astype(np.float32)

                # Map local point indices to global 3D point ids
                local_to_global = []
                for i, (pid, pt3d) in enumerate(zip(valid_p3d_ids, db_pts3d)):
                    if pid not in pts3d_global_dict:
                        pts3d_global_dict[pid] = (pt3d, global_idx, db_pts3d_color_norm[i])
                        global_idx += 1
                    local_to_global.append(pts3d_global_dict[pid][1])

                # Down-weight lower-ranked retrieved images
                weight = np.exp(-rank * 0.05)

                covis_data.append({
                    "local_to_global": np.array(local_to_global),
                    "pts3dm": db_pts3dm,
                    "color3d": db_pts3d_color_norm,
                    "weight": weight,
                })

            if len(pts3d_global_dict) == 0:
                continue

            pts3d_global_arr = np.zeros((len(pts3d_global_dict), 3), dtype=np.float32)
            for pid, data in pts3d_global_dict.items():
                pts3d_global_arr[data[1]] = data[0]

            matches_scores = np.zeros((len(pts3d_global_dict), len(pts2d)), dtype=np.float32)

            match_start_time = time.time()

            # Match per retrieved image and merge scores
            for covis in covis_data:
                local_to_global = covis["local_to_global"]
                pts3dm_covis = covis["pts3dm"]
                color3d_covis = covis["color3d"]
                weight = covis["weight"]

                max_pts3d_per_covis = 1024
                if len(pts3dm_covis) > max_pts3d_per_covis:
                    indices = np.random.choice(len(pts3dm_covis), max_pts3d_per_covis, replace=False)
                    pts3dm_covis = pts3dm_covis[indices]
                    local_to_global = local_to_global[indices]
                    color3d_covis = color3d_covis[indices]

                pts2d_t, idx2d_t, pts3dm_t, idx3d_t, color2d_t, color3d_t = prepare_sample_inputs(
                    pts2d, pts3dm_covis, color2d, color3d_covis, device)
                _, match_probs_list = matcher(pts2d_t, idx2d_t, pts3dm_t, idx3d_t, color2d_t, color3d_t)

                match_probs = match_probs_list[0].cpu().numpy()
                i3d_local, i2d = np.where(match_probs[:-1, :-1] > sc_thres)
                for i3d_l, i2d_l in zip(i3d_local, i2d):
                    i3d_g = local_to_global[i3d_l]
                    score = match_probs[i3d_l, i2d_l] * weight
                    matches_scores[i3d_g, i2d_l] = max(matches_scores[i3d_g, i2d_l], score)

            i3d, i2d = np.where(matches_scores > sc_thres)
            if len(i3d) < 4:
                continue

            matching_times.append(time.time() - match_start_time)

            pose_result = estimate_pose(
                pts2d[i2d], pts3d_global_arr[i3d],
                ransac_thres=0.001, iterations_count=1000, confidence=0.99,
            )
            if pose_result is not None:
                R, t, _ = pose_result
                results_list.append({
                    "name": query_name,
                    "qvec": rotation_matrix_to_quaternion(R),
                    "tvec": t,
                })

    # Save predicted poses (visuallocalization.net submission format)
    with open(results_path, "w") as f:
        for r in results_list:
            qvec, tvec = r["qvec"], r["tvec"]
            img_name = os.path.basename(r["name"])
            f.write(f"{img_name} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                    f"{tvec[0]} {tvec[1]} {tvec[2]}\n")

    print(f"\nResults saved to: {results_path}")
    print(f"Successfully localized: {len(results_list)}/{len(all_queries)} images")
    if matching_times:
        print(f"Average matching time per image (excluding PnP): {np.mean(matching_times):.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aachen Day-Night localization with GeoMix")
    parser.add_argument("--dataset_path", type=str, default="data/aachen",
                        help="Path to the Aachen Day-Night dataset root")
    parser.add_argument("--outputs", type=str, default="outputs/aachen",
                        help="Output directory")
    parser.add_argument("--weights", type=str, default="geomix_best.ckpt",
                        help="Path to the GeoMix checkpoint")
    parser.add_argument("--detector_2d", type=str, default="sift",
                        choices=["sift", "superpoint", "r2d2", "disk"],
                        help="2D (query-side) feature detector")
    parser.add_argument("--detector_3d", type=str, default="sift",
                        choices=["sift", "superpoint", "r2d2", "disk"],
                        help="3D (map-side) feature detector used for retriangulation")
    parser.add_argument("--sfm_pairs", type=str, default="data/aachen/pairs/pairs-db-covis20.txt",
                        help="Database covisibility pairs for triangulation (from hloc pairs/aachen)")
    parser.add_argument("--loc_pairs", type=str, default="data/aachen/pairs/pairs-query-netvlad50.txt",
                        help="Query-database retrieval pairs (from hloc pairs/aachen)")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of retrieved database images per query")

    main(parser.parse_args())
