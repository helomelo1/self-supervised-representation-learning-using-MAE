---
title: MAE CIFAR-10 Demo
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Self-Supervised Learning with Masked Autoencoders (MAE)

Implemented a **Masked Autoencoder** from scratch in PyTorch based on [He et al., 2021](https://arxiv.org/abs/2111.06377), demonstrating self-supervised visual representation learning on CIFAR-10 — deployed as a **live inference API**.

> **[Try the Live Demo →](https://shrihari8-mae-cifar10.hf.space/docs)**

## Key Highlights

- **Built Vision Transformer architecture from scratch** — implemented multi-head self-attention, positional embeddings, and encoder-decoder modules without using pre-built ViT libraries
- **Achieved efficient self-supervised pre-training** — trained with 75% patch masking, reducing compute while learning robust representations
- **Validated learned representations** — evaluated feature quality using linear probing and t-SNE visualization
- **Deployed as a REST API** — FastAPI inference server containerized with Docker, hosted on Hugging Face Spaces

## Technical Implementation

| Component | Details |
|-----------|---------|
| **Framework** | PyTorch |
| **Architecture** | Vision Transformer (ViT) |
| **Encoder** | 6 transformer blocks, 192-dim embeddings, 3 attention heads |
| **Decoder** | 6 transformer blocks, 128-dim embeddings, 4 attention heads |
| **Input Processing** | 4×4 patches (64 patches per 32×32 image) |
| **Masking Strategy** | Random 75% patch masking with MSE reconstruction loss |
| **Deployment** | FastAPI + Docker on Hugging Face Spaces |

## Results

| Evaluation Method | Accuracy |
|-------------------|----------|
| **Linear Probing (MAE features)** | 61.25% |
| **k-NN Classification (MAE features)** | 50.4% |
| Raw Pixels (baseline) | 34.63% |

## Learned Representations (t-SNE)

The encoder learns semantically meaningful features — visualized below with t-SNE showing clear class separation after self-supervised pre-training:

![t-SNE Visualization](images/tsne.png)

## How It Works

```
Image → Patchify (4×4) → Mask 75% → Encoder (visible patches) → Decoder → Reconstruct masked patches
```

The model learns by predicting missing pixel values from visible context, forcing the encoder to capture high-level semantic features.

## API Endpoints

The trained model is served via a FastAPI inference API with three endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | `POST` | Upload an image → get top-5 CIFAR-10 class predictions |
| `/reconstruct` | `POST` | Upload an image → get original, masked, and reconstructed images (base64 PNG) |
| `/embed` | `POST` | Upload an image → get the 192-dim encoder feature vector |
| `/health` | `GET` | Health check |

### Example Usage

```bash
# Classify an image
curl -X POST https://shrihari8-mae-cifar10.hf.space/classify \
  -F "file=@your_image.png"

# Get encoder embeddings
curl -X POST https://shrihari8-mae-cifar10.hf.space/embed \
  -F "file=@your_image.png"

# Reconstruct masked image
curl -X POST https://shrihari8-mae-cifar10.hf.space/reconstruct \
  -F "file=@your_image.png"
```

## Local Development

```bash
# Pre-train MAE
python train.py

# Linear probe evaluation
python probe.py

# Generate t-SNE visualization
python tsne.py

# Run API locally with Docker
docker compose up -d
# API available at http://localhost:8000/docs
```

## Deployment

The API is containerized with Docker and deployed on [Hugging Face Spaces](https://huggingface.co/spaces/shrihari8/mae-cifar10):

```bash
# Build and run locally
docker compose up -d

# Deploy to Hugging Face Spaces
git push hf main
```

## Skills Demonstrated

`PyTorch` · `Vision Transformers` · `Self-Supervised Learning` · `Attention Mechanisms` · `Representation Learning` · `FastAPI` · `Docker` · `MLOps`

## Reference

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021). *Masked Autoencoders Are Scalable Vision Learners*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
