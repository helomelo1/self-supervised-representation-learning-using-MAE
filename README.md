# Self-Supervised Learning with Masked Autoencoders (MAE)

Implemented a **Masked Autoencoder** from scratch in PyTorch based on [He et al., 2021](https://arxiv.org/abs/2111.06377), demonstrating self-supervised visual representation learning on CIFAR-10.

## Key Highlights

- **Built Vision Transformer architecture from scratch** — implemented multi-head self-attention, positional embeddings, and encoder-decoder modules without using pre-built ViT libraries
- **Achieved efficient self-supervised pre-training** — trained with 75% patch masking, reducing compute while learning robust representations
- **Validated learned representations** — evaluated feature quality using linear probing and t-SNE visualization

## Technical Implementation

| Component | Details |
|-----------|---------|
| **Framework** | PyTorch |
| **Architecture** | Vision Transformer (ViT) |
| **Encoder** | 6 transformer blocks, 192-dim embeddings, 3 attention heads |
| **Decoder** | 6 transformer blocks, 128-dim embeddings, 4 attention heads |
| **Input Processing** | 4×4 patches (64 patches per 32×32 image) |
| **Masking Strategy** | Random 75% patch masking with MSE reconstruction loss |

## Results

| Evaluation Method | Accuracy |
|-------------------|----------|
| **Linear Probing** | 61.25% |
| **k-NN Classification** | 50.4% |

## Learned Representations (t-SNE)

The encoder learns semantically meaningful features — visualized below with t-SNE showing clear class separation after self-supervised pre-training:

![t-SNE Visualization](images/tsne.png)

## How It Works

```
Image → Patchify (4×4) → Mask 75% → Encoder (visible patches) → Decoder → Reconstruct masked patches
```

The model learns by predicting missing pixel values from visible context, forcing the encoder to capture high-level semantic features.

## Usage

```bash
# Pre-train MAE
python train.py

# Linear probe evaluation
python probe.py

# Generate t-SNE visualization
python tsne.py
```

## Skills Demonstrated

`PyTorch` · `Vision Transformers` · `Self-Supervised Learning` · `Attention Mechanisms` · `Representation Learning`

## Reference

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021). *Masked Autoencoders Are Scalable Vision Learners*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
