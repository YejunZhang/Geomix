# [ECCV 2026] GeoMix: Descriptor-Free Visual Localization via Global Context and Multi-Detector Training

Authors: [Yejun Zhang](https://yejunzhang.github.io), [Xinjue Wang](https://xnnjw.github.io/), Zihan Wang, [Esa Rahtu](https://esa.rahtu.fi/), and [Juho Kannala](https://users.aalto.fi/~kannalj1/)

[[arXiv](https://arxiv.org/abs/2607.02486)]

GeoMix is a descriptor-free 2D-3D matching framework for visual localization, built on [A2-GNN](https://github.com/YejunZhang/a2-gnn) with directional and distance-aware local embeddings, learnable global context nodes, and Mix-Training over multiple keypoint detectors (SIFT + SuperPoint + DISK).

![GeoMix pipeline](assets/pipeline.png)

## Environment Setup

```
git clone https://github.com/YejunZhang/Geomix.git
cd Geomix
conda env create -f environment.yml
conda activate geomix

wget https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_scatter-2.0.8-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.8-cp37-cp37m-linux_x86_64.whl
pip install . --find-links https://data.pyg.org/whl/torch-1.8.0+cu11.1.html
```

## Data Preparation

See [tools/README.md](tools/README.md) for downloading the datasets and generating the multi-detector keypoint caches.

## Training & Evaluation

The pretrained Mix-Training model is included as ```geomix_best.ckpt```.

```
# Train on MegaDepth (Mix-Training: SIFT + SuperPoint + DISK)
sh train.sh

# Eval on MegaDepth with each detector
sh eval.sh
```

Visual localization on Cambridge Landmarks / 7Scenes uses the same entrypoint, with `--dataset` in `{megadepth, cambridge_sift, 7scenes_sift_v2, 7scenes_superpoint_v2}`:

```
python -m geomix_eval.benchmark --root_dir . --ckpt geomix_best.ckpt \
    --dataset cambridge_sift --splits kings --p2d_type superpoint \
    --covis_k_nums 10 --odir outputs/eval/cambridge
```

Aachen Day-Night uses an [hloc](https://github.com/cvg/Hierarchical-Localization)-based pipeline that retriangulates the 3D model with the chosen detector and writes poses in the [visuallocalization.net](https://www.visuallocalization.net/) format:

```
python eval_aachen.py --detector_2d superpoint --detector_3d superpoint
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

We appreciate the previous open-source repository [GoMatch](https://github.com/dvl-tum/gomatch), [DGC-GNN](https://github.com/AaltoVision/DGC-GNN-release), [A2-GNN](https://github.com/YejunZhang/a2-gnn) and [CLNet](https://github.com/sailor-z/CLNet).

## Citation

Please consider citing our papers if you find this code useful for your research:

```
@inproceedings{zhang2026geomix,
      title={GeoMix: Descriptor-Free Visual Localization via Global Context and Multi-Detector Training},
      author={Yejun Zhang and Xinjue Wang and Zihan Wang and Esa Rahtu and Juho Kannala},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2026},
}

@inproceedings{zhang2025a2gnn,
      title={A2-GNN: Angle-Annular GNN for Visual Descriptor-free Camera Relocalization},
      author={Yejun Zhang and Shuzhe Wang and Juho Kannala},
      booktitle={International Conference on 3D Vision (3DV)},
      year={2025},
}
```
