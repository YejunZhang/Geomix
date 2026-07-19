# Data Preparation Tools

`extract_features.py` generates the per-detector keypoint caches
(`desc_cache/<DIR>/<scene>.npy`, entries `{kpts, descs, color}`) on top of the
DGC-GNN / GoMatch processed data. GeoMix only uses keypoints and colors, so
descriptors are stored as placeholders (`--save_descs` keeps real ones). The
top 1024 keypoints by score are kept; already-cached images are skipped.

| `--detector` | cache dir | backend |
|---|---|---|
| `superpoint` | `SuperPoint_r4` | [HF SuperPoint](https://huggingface.co/magic-leap-community/superpoint), thr 0.005, NMS r=4 |
| `disk` | `disk` | [kornia](https://github.com/kornia/kornia) `DISK.from_pretrained` |
| `r2d2` | `r2d2` | [R2D2](https://github.com/naver/r2d2), `r2d2_WASF_N16.pt`, NMS 0.7/0.7 |
| `dedode` | `dedode` | [DeDoDe](https://github.com/Parskatt/DeDoDe), detector L v2 |

```
# Mix-Training caches
python tools/extract_features.py --detector superpoint --splits train val test
python tools/extract_features.py --detector disk --splits train val test

# Zero-shot evaluation caches
python tools/extract_features.py --detector r2d2 --splits test \
    --r2d2_repo /path/to/r2d2 --r2d2_ckpt /path/to/r2d2_WASF_N16.pt
python tools/extract_features.py --detector dedode --splits test \
    --dedode_repo /path/to/DeDoDe --dedode_ckpt /path/to/dedode_detector_L_v2.pth
```

MegaDepth images come from the [D2-Net preprocessing](https://github.com/mihaidusmanu/d2-net#downloading-and-preprocessing-the-megadepth-dataset); use `--im_dir` if they live outside `data/`.

**Environment:** use a recent Python/torch environment (not the `geomix` training env) with your detector's dependency (`transformers>=4.39` / `kornia>=0.6.7` / the official repos), and pin `numpy<2` so the caches stay readable by the NumPy 1.x training env.
