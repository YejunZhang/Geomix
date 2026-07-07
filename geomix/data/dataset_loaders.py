from argparse import Namespace
from typing import Any, Dict, Iterable, Mapping, Tuple, Union

import torch
from torch.utils.data import DataLoader,ConcatDataset

from .datasets import BaseDataset, SIFTDataset, SuperPointDataset, DISKDataset
from ..utils.logger import get_logger

_logger = get_logger(level="INFO", name="data_loader")


def collate(all_data: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    # Ignore samples with no pts
    data = []
    for d in all_data:
        if len(d["pts2d"]) > 0:
            data.append(d)
    if len(data) == 0:
        batched: Dict[str, Any] = dict(name=None)
        return batched

    # Batch data contents
    batched = dict(
        name=[d["name"] for d in data],
        pts2d=torch.cat([torch.from_numpy(d["pts2d"]) for d in data]),
        pts3d=torch.cat([torch.from_numpy(d["pts3d"]) for d in data]),
        pts2d_pix=torch.cat([torch.from_numpy(d["pts2d_pix"]) for d in data]),
        pts2dm=torch.cat([torch.from_numpy(d["pts2dm"]) for d in data]),
        pts3dm=torch.cat([torch.from_numpy(d["pts3dm"]) for d in data]),
        color2d = torch.cat([torch.from_numpy(d["color2d"]) for d in data]),
        color3d = torch.cat([torch.from_numpy(d["color3d"]) for d in data]),
        idx2d=torch.cat(
            [
                torch.full((len(d["pts2d"]),), i, dtype=torch.long)
                for i, d in enumerate(data)
            ]
        ),
        idx3d=torch.cat(
            [
                torch.full((len(d["pts3d"]),), i, dtype=torch.long)
                for i, d in enumerate(data)
            ]
        ),
        matches_bin=torch.cat(
            [torch.from_numpy(d["matches_bin"]).view(-1) for d in data]
        ),
        R=torch.stack([torch.from_numpy(d["R"]) for d in data]),
        t=torch.stack([torch.from_numpy(d["t"]) for d in data]),
        K=torch.stack([torch.from_numpy(d["K"]) for d in data]),
    )

    # Special data for multi-covis evaluation
    if "unmerge_mask" in data[0]:
        batched["unmerge_mask"] = [d["unmerge_mask"] for d in data]
        batched["idx3dm"] = torch.cat(
            [
                torch.full((len(d["pts3dm"]),), i, dtype=torch.long)
                for i, d in enumerate(data)
            ]
        )
    if "covis_ids" in data[0]:
        batched["covis_ids"] = [d["covis_ids"] for d in data]
        
    # import pdb
    # pdb.set_trace()
    return batched


def init_data_loader(
    config: Namespace,
    split: str = "train",
    batch: int = 16,
    overfit: int = -1,
    outlier_rate: Union[float, Tuple[float, float], None] = None,
    npts: Union[int, Tuple[int, int], None] = None,
    dataset_class: type = BaseDataset,
) -> DataLoader:
    is_training = "train" in split
    is_evaluation = "val" in split
    batch = batch if "batch" not in config else config.batch
    num_workers = 0 if "num_workers" not in config else config.num_workers
    _logger.info(
        f"Init data loader: split={split} training={is_training} batch={batch}..."
    )

    # Load datasets - create separate dataset for each detector type with their specific topk
    from copy import copy

    detector_configs = {
        "sift": {"topk": 1, "class": SIFTDataset},
        "superpoint": {"topk": 3, "class": SuperPointDataset},
        "disk": {"topk": 3, "class": DISKDataset},
    }

    # Select detectors from config (default: all three)
    detectors = getattr(config, "detectors", ["sift", "superpoint", "disk"])

    datasets = []
    for det in detectors:
        cfg = copy(config)
        cfg.topk = detector_configs[det]["topk"]
        cfg.p2d_type = det
        ds = detector_configs[det]["class"](cfg, split=split)
        if outlier_rate is not None:
            ds.outlier_rate = outlier_rate
        if npts is not None:
            ds.npts = npts
        _logger.info(ds)
        datasets.append(ds)

    dataset = ConcatDataset(datasets)
    det_names = "+".join([d.upper() for d in detectors])
    _logger.info(f"Combined {len(detectors)} detectors ({det_names}), total samples: {len(dataset)}")

    if overfit > 0:
        dataset, _ = torch.utils.data.random_split(
            dataset, [overfit, len(dataset) - overfit]
        )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        num_workers=num_workers,
        collate_fn=collate,
        shuffle=is_training,
        drop_last=is_training,
    )
    return data_loader


def init_eval_data_loader(
    config: Namespace,
    split: str = "test",
    batch: int = 16,
    p2d_type: str = "sift",
    overfit: int = -1,
    outlier_rate: Union[float, Tuple[float, float], None] = None,
    npts: Union[int, Tuple[int, int], None] = None,
) -> DataLoader:
    """Initialize data loader for evaluation with a single detector type."""
    is_training = "train" in split
    batch = batch if "batch" not in config else config.batch
    num_workers = 0 if "num_workers" not in config else config.num_workers
    _logger.info(
        f"Init eval data loader: split={split} p2d_type={p2d_type} batch={batch}..."
    )

    # Detector-specific topk values
    detector_topks = {
        "sift": 1,
        "superpoint": 3,
        "disk": 3,
    }

    # Create config copy with detector-specific topk
    from copy import copy
    config_detector = copy(config)
    config_detector.topk = detector_topks.get(p2d_type, 1)
    config_detector.p2d_type = p2d_type

    # Load the specified detector dataset
    detector_classes = {
        "sift": SIFTDataset,
        "superpoint": SuperPointDataset,
        "disk": DISKDataset,
    }

    dataset_class = detector_classes.get(p2d_type, SIFTDataset)
    dataset = dataset_class(config_detector, split=split)

    # Apply config overrides
    if outlier_rate is not None:
        dataset.outlier_rate = outlier_rate
    if npts is not None:
        dataset.npts = npts
    _logger.info(dataset)

    if overfit > 0:
        dataset, _ = torch.utils.data.random_split(
            dataset, [overfit, len(dataset) - overfit]
        )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        num_workers=num_workers,
        collate_fn=collate,
        shuffle=is_training,
        drop_last=is_training,
    )
    return data_loader
