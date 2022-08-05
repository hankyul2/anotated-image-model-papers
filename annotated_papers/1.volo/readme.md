# VOLO

This document contains about how to implement volo architecture.

## Tutorial

1. Check model architecture

```bash
python3 test_volo.py
```

2. Train VOLO D1

```bash
python3 train.py data --dataset_type CIFAR100 --model-name volo_d1_224 --train-size 224 224  --train-resize-mode ResizeRandomCrop --random-crop-pad 28 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.4914 0.4825 0.4467 --std 0.2471 0.2435 0.2616 --cutmix 1.0 --mixup 0.0 --remode 0.0 --drop-path-rate 0.0 --smoothing 0.0 --epoch 300 --optimizer sgd --nesterov --lr 0.25 --min-lr 1e-4 --weight-decay 1e-4 --warmup-epoch 5 --scheduler cosine -b 128 -j 4 --pin-memory --amp --channels-last --cuda 8
```

## Train Results
