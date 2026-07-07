#!/bin/bash
# Train GeoMix on MegaDepth with Mix-Training (SIFT + SuperPoint + DISK, default).

python -m geomix_train.train_matcher --gpus 0 --batch 16 -lr 0.0005 \
    --max_epochs 50 --matcher_class 'OTMatcherCls' --share_kp2d_enc \
    --dataset 'megadepth' --train_split 'train' --val_split 'val' \
    --outlier_rate 0.5 0.5 --npts 100 1024 \
    --p3d_type 'bvs' \
    --inls2d_thres 0.001 --rpthres 0.01 \
    -o 'geomix_mix_training'

# Single-detector training baseline (SIFT only):
# python -m geomix_train.train_matcher --gpus 0 --batch 16 -lr 0.0005 \
#     --max_epochs 50 --matcher_class 'OTMatcherCls' --share_kp2d_enc \
#     --dataset 'megadepth' --train_split 'train' --val_split 'val' \
#     --outlier_rate 0.5 0.5 --topk 1 --npts 100 1024 \
#     --p2d_type 'sift' --p3d_type 'bvs' \
#     --inls2d_thres 0.001 --rpthres 0.01 \
#     -o 'geomix_single_sift'
