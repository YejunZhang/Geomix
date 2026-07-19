"""Unified keypoint-cache extraction for GeoMix.

Generates the per-detector keypoint caches consumed by the GeoMix data
pipeline (``desc_cache/<DIR>/<scene>.npy``) on top of the processed scene
files released by DGC-GNN / produced by the GoMatch toolchain
(``<dataset>_2d3d*.npy``). One cache entry per image:

    {img_id: {"kpts": (N, 2) float32, "descs": ..., "color": (N, 3) uint8}}

GeoMix is descriptor-free: only ``kpts`` and ``color`` are used downstream,
so descriptors are stored as small placeholders by default (``--save_descs``
keeps the real ones, only needed for descriptor-based baselines).

Supported detectors and their cache directory names (must match
``geomix.data.datasets.FEATURE_DIRS``):

    superpoint -> SuperPoint_r4   (HF magic-leap-community/superpoint,
                                   keypoint_threshold=0.005, nms_radius=4)
    disk       -> disk            (kornia DISK.from_pretrained)
    r2d2       -> r2d2            (official R2D2 repo, r2d2_WASF_N16.pt,
                                   NMS rel_thr=rep_thr=0.7)
    dedode     -> dedode          (official DeDoDe repo, detector L v2)

All detectors keep the top ``--max_keypoints`` (default 1024) keypoints by
detection score. Extraction is resume-safe: images already present in an
existing cache file are skipped.

Examples (run from the repository root, data under ./data):

    # Mix-Training caches (all splits)
    python tools/extract_features.py --detector superpoint --splits train val test
    python tools/extract_features.py --detector disk --splits train val test

    # Zero-shot evaluation caches (test split only)
    python tools/extract_features.py --detector r2d2 --splits test \
        --r2d2_repo /path/to/r2d2 --r2d2_ckpt /path/to/r2d2/models/r2d2_WASF_N16.pt
    python tools/extract_features.py --detector dedode --splits test \
        --dedode_repo /path/to/DeDoDe --dedode_ckpt /path/to/dedode_detector_L_v2.pth
"""

import argparse
from argparse import Namespace
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Cache dir names expected by geomix.data.datasets.FEATURE_DIRS
FEATURE_DIRS = {
    "superpoint": "SuperPoint_r4",
    "disk": "disk",
    "r2d2": "r2d2",
    "dedode": "dedode",
}

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_keypoint_colors(img_rgb, kpts):
    """RGB color at each (rounded, clamped) keypoint location."""
    xs = np.clip(kpts[:, 0].astype(int), 0, img_rgb.shape[1] - 1)
    ys = np.clip(kpts[:, 1].astype(int), 0, img_rgb.shape[0] - 1)
    return img_rgb[ys, xs].astype(np.uint8)


class SuperPointExtractor:
    """HF SuperPoint: keypoint_threshold=0.005, nms_radius=4, top-k by score."""

    def __init__(self, args):
        from transformers import AutoImageProcessor, SuperPointForKeypointDetection

        self.threshold = 0.005
        self.processor = AutoImageProcessor.from_pretrained(
            "magic-leap-community/superpoint",
            keypoint_threshold=self.threshold,
            nms_radius=4,
        )
        self.model = SuperPointForKeypointDetection.from_pretrained(
            "magic-leap-community/superpoint"
        ).to(DEV).eval()
        self.max_keypoints = args.max_keypoints

    def __call__(self, im_path, img_rgb):
        from PIL import Image

        pil_image = Image.fromarray(img_rgb)
        inputs = self.processor(pil_image, return_tensors="pt").to(DEV)
        with torch.no_grad():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_keypoint_detection(
            outputs, [(pil_image.height, pil_image.width)]
        )
        kpts = processed[0]["keypoints"].cpu().numpy()
        scores = processed[0]["scores"].cpu().numpy()

        mask = scores >= self.threshold
        kpts, scores = kpts[mask], scores[mask]
        order = np.argsort(scores)[::-1][: self.max_keypoints]
        # Placeholder descriptors, matching the released SuperPoint_r4 cache
        return kpts[order].astype(np.float32), np.array([0.0, 1.0])


class DiskExtractor:
    """Kornia DISK: pad to /16, top-k keypoints by detection score."""

    def __init__(self, args):
        import kornia as K

        self.K = K
        self.model = K.feature.DISK.from_pretrained(device=DEV)
        self.max_keypoints = args.max_keypoints
        self.save_descs = args.save_descs

    def __call__(self, im_path, img_rgb):
        import torch.nn.functional as F

        img_tensor = self.K.utils.image_to_tensor(img_rgb).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        _, _, h, w = img_tensor.shape
        pad_h = (16 - h % 16) if h % 16 != 0 else 0
        pad_w = (16 - w % 16) if w % 16 != 0 else 0
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

        with torch.no_grad():
            feats = self.model(img_tensor.to(DEV))[0]
        top_k = min(self.max_keypoints, len(feats.detection_scores))
        idxs = torch.topk(feats.detection_scores, top_k).indices
        kpts = feats.keypoints[idxs].cpu().numpy().astype(np.float32)
        if self.save_descs:
            descs = feats.descriptors[idxs].cpu().numpy()
        else:
            descs = np.zeros((len(kpts), 1), dtype=np.float16)
        return kpts, descs


class R2d2Extractor:
    """Official R2D2: NMS rel_thr=rep_thr=0.7, top-k by reliability*repeatability."""

    def __init__(self, args):
        if not args.r2d2_repo or not args.r2d2_ckpt:
            raise SystemExit("--r2d2_repo and --r2d2_ckpt are required for r2d2")
        sys.path.insert(0, os.path.abspath(args.r2d2_repo))
        from tools.dataloader import norm_RGB  # noqa: F401 (r2d2 repo module)
        import nets.patchnet as patchnet

        self.norm_RGB = norm_RGB
        checkpoint = torch.load(args.r2d2_ckpt, map_location="cpu")
        print(">> Creating net = " + checkpoint["net"])
        net = eval(checkpoint["net"], vars(patchnet))
        weights = checkpoint["state_dict"]
        net.load_state_dict({k.replace("module.", ""): v for k, v in weights.items()})
        self.net = net.eval().to(DEV)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = 0.7
        self.rep_thr = 0.7
        self.max_keypoints = args.max_keypoints
        self.save_descs = args.save_descs

    def __call__(self, im_path, img_rgb):
        from PIL import Image

        img = Image.fromarray(img_rgb)
        img_tensor = self.norm_RGB(img)[None].to(DEV)
        with torch.no_grad():
            res = self.net(imgs=[img_tensor])

        descriptors = res["descriptors"][0]
        reliability = res["reliability"][0]
        repeatability = res["repeatability"][0]

        # non-maximum suppression on the repeatability map
        maxima = repeatability == self.max_filter(repeatability)
        maxima *= repeatability >= self.rep_thr
        maxima *= reliability >= self.rel_thr
        y, x = maxima.nonzero().t()[2:4]

        c = reliability[0, 0, y, x]
        q = repeatability[0, 0, y, x]
        d = descriptors[0, :, y, x].t()
        scores = c * q
        if len(scores) > self.max_keypoints:
            idxs = scores.argsort(descending=True)[: self.max_keypoints]
            x, y, d = x[idxs], y[idxs], d[idxs]

        kpts = torch.stack([x.float(), y.float()], dim=-1).cpu().numpy().astype(np.float32)
        if self.save_descs:
            descs = d.cpu().numpy()
        else:
            descs = np.zeros((len(kpts), 1), dtype=np.float16)
        return kpts, descs


class DedodeExtractor:
    """Official DeDoDe detector L (v2 weights)."""

    def __init__(self, args):
        if not args.dedode_repo or not args.dedode_ckpt:
            raise SystemExit("--dedode_repo and --dedode_ckpt are required for dedode")
        sys.path.insert(0, os.path.abspath(args.dedode_repo))
        from DeDoDe import dedode_detector_L

        weights = torch.load(args.dedode_ckpt, map_location=DEV)
        self.model = dedode_detector_L(weights=weights).to(DEV)
        self.max_keypoints = args.max_keypoints

    def __call__(self, im_path, img_rgb):
        h, w = img_rgb.shape[:2]
        with torch.no_grad():
            out = self.model.detect_from_path(im_path, num_keypoints=self.max_keypoints)
        kpts = self.model.to_pixel_coords(out["keypoints"], h, w)[0]
        kpts = kpts.cpu().numpy().astype(np.float32)
        descs = np.zeros((len(kpts), 1), dtype=np.float16)
        return kpts, descs


EXTRACTORS = {
    "superpoint": SuperPointExtractor,
    "disk": DiskExtractor,
    "r2d2": R2d2Extractor,
    "dedode": DedodeExtractor,
}


def compute_scene_im_features(args, extractor, split):
    with open(args.dataset_config, "r") as f:
        dataset_conf = Namespace(**yaml.load(f, Loader=yaml.FullLoader)[args.dataset])

    data_root = Path(args.root_dir) / "data"
    if args.im_dir:
        im_dir = Path(args.im_dir)
    elif str(dataset_conf.data_dir).startswith("/"):
        im_dir = Path(dataset_conf.data_dir)
    else:
        im_dir = data_root / dataset_conf.data_dir

    data_processed_dir = data_root / dataset_conf.data_processed_dir
    data_file = data_processed_dir / dataset_conf.data_file
    sids_to_load = dataset_conf.splits[split]

    feature_cache_dir = data_processed_dir / "desc_cache" / FEATURE_DIRS[args.detector]
    feature_cache_dir.mkdir(exist_ok=True, parents=True)

    print(f">>>> Loading data from {data_file}")
    data_dict = np.load(data_file, allow_pickle=True).item()
    print(f"Extract features per scene, detector={args.detector} cache dir={feature_cache_dir}")

    for sid in tqdm(sids_to_load, total=len(sids_to_load)):
        if sid not in data_dict:
            continue

        feature_path = feature_cache_dir / f"{sid}.npy"
        scene_ims = data_dict[sid]["ims"]
        if feature_path.exists():
            scene_features = np.load(feature_path, allow_pickle=True).item()
        else:
            scene_features = {}
        print(f"sid={sid} ims={len(scene_ims)} cached={len(scene_features)}")

        updated_count = 0
        for imid, im in scene_ims.items():
            if imid in scene_features:
                continue
            im_path = os.path.join(im_dir, im.name)
            img = cv2.imread(im_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image at {im_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            kpts, descs = extractor(im_path, img_rgb)
            color = sample_keypoint_colors(img_rgb, kpts)
            scene_features[imid] = {"kpts": kpts, "descs": descs, "color": color}
            updated_count += 1

        if updated_count > 0:
            print(f"Save {updated_count} new image features.")
            np.save(feature_path, scene_features)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detector", type=str, required=True, choices=list(EXTRACTORS))
    parser.add_argument("--dataset", type=str, default="megadepth")
    parser.add_argument("--dataset_config", type=str, default="configs/datasets.yml")
    parser.add_argument("--splits", type=str, nargs="+", default=["test"],
                        help="e.g. train val test (superpoint/disk) or test (r2d2/dedode)")
    parser.add_argument("--root_dir", type=str, default=".",
                        help="repo root; data is expected under <root_dir>/data")
    parser.add_argument("--im_dir", type=str, default=None,
                        help="override the image directory from the dataset config")
    parser.add_argument("--max_keypoints", type=int, default=1024)
    parser.add_argument("--save_descs", action="store_true",
                        help="store real descriptors (disk/r2d2). GeoMix does not need "
                             "them; default stores small placeholders to save space")
    # external repos/checkpoints for detectors without pip packages
    parser.add_argument("--r2d2_repo", type=str, default=None,
                        help="path to a clone of github.com/naver/r2d2")
    parser.add_argument("--r2d2_ckpt", type=str, default=None,
                        help="path to r2d2_WASF_N16.pt")
    parser.add_argument("--dedode_repo", type=str, default=None,
                        help="path to a clone of github.com/Parskatt/DeDoDe")
    parser.add_argument("--dedode_ckpt", type=str, default=None,
                        help="path to dedode_detector_L_v2.pth")
    args = parser.parse_args()
    print(args)

    extractor = EXTRACTORS[args.detector](args)
    for split in args.splits:
        compute_scene_im_features(args, extractor, split)


if __name__ == "__main__":
    main()
