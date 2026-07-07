#!/bin/bash
# Evaluate the released mix-training model on MegaDepth (test split)
# with each keypoint detector (SIFT / SuperPoint / DISK).

CKPT="geomix_best.ckpt"
OUTPUT_ROOT="outputs/eval/megadepth"

DETECTORS=("sift" "superpoint" "disk")

for DETECTOR in "${DETECTORS[@]}"; do
    echo "Evaluating $CKPT with detector: $DETECTOR ..."
    mkdir -p "${OUTPUT_ROOT}/${DETECTOR}"

    python -m geomix_eval.benchmark \
        --root_dir . \
        --ckpt "$CKPT" \
        --splits 'test' \
        --odir "${OUTPUT_ROOT}/${DETECTOR}" \
        --dataset 'megadepth' \
        --covis_k_nums 10 \
        --p2d_type "${DETECTOR}"

    echo "Finished eval with $DETECTOR"
done
