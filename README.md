# Masked Autoencoder (MAE) for Representation Learning

A PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) for self-supervised representation learning on CIFAR-10.

## Overview

MAE learns visual representations by masking 75% of image patches and training a Vision Transformer to reconstruct the missing pixels.

```
Input Image → Patchify → Mask 75% → Encoder (visible only) → Decoder → Reconstruct
```

## Project Structure

- `data.py` — Data loading, augmentation, patchification, and masking
- `model.py` — MAE architecture (encoder + decoder)
- `train.py` — Pre-training script
- `probe.py` — Linear probe evaluation

## Quick Start

```bash
# Pre-train MAE
python train.py

# Evaluate with linear probe
python probe.py
```

## Architecture

| Component | Config |
|-----------|--------|
| Patch size | 4×4 (64 patches per image) |
| Mask ratio | 75% |
| Encoder | 6 ViT blocks, dim=192, heads=3 |
| Decoder | 6 ViT blocks, dim=128, heads=4 |

## References

- [MAE Paper](https://arxiv.org/abs/2111.06377) — He et al., 2021