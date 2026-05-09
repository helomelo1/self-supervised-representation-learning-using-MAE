import io
import base64
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torchvision.transforms as transforms

from model import MAE
from data import patchify, random_masking, unpatchify, CIFAR10_MEAN, CIFAR10_STD

# ─── Config ───────────────────────────────────────────────────────────────────

DEVICE = "cpu"
PATCH_SIZE = 4
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3
NUM_PATCHES = (32 // PATCH_SIZE) ** 2
EMBED_DIM = 192

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ─── Model Loading ────────────────────────────────────────────────────────────

mae = MAE(patch_dim=PATCH_DIM, num_patches=NUM_PATCHES).to(DEVICE)
mae.load_state_dict(torch.load("weights/mae_cifar10_150ep.pth", map_location=DEVICE))
mae.eval()

encoder = mae.encoder

classifier = nn.Linear(EMBED_DIM, 10).to(DEVICE)
classifier.load_state_dict(torch.load("weights/probe.pth", map_location=DEVICE))
classifier.eval()

# ─── Preprocessing ────────────────────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def load_image(file_bytes: bytes) -> torch.Tensor:
    """Load uploaded image bytes into a preprocessed (1, C, H, W) tensor."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)  # (1, 3, 32, 32)
    return tensor


def encode(image_tensor: torch.Tensor) -> torch.Tensor:
    """Run encoder on all patches (no masking) → (1, NUM_PATCHES, EMBED_DIM)."""
    patches = patchify(image_tensor)
    ids_keep = torch.arange(NUM_PATCHES).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = encoder(patches, ids_keep)

    return features


def tensor_to_base64_png(tensor: torch.Tensor) -> str:
    """Convert a (C, H, W) image tensor to a base64-encoded PNG string.
    Denormalizes using CIFAR-10 stats before encoding."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)

    image = transforms.ToPILImage()(tensor)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="MAE Inference API",
    description="Self-supervised Masked Autoencoder (MAE) trained on CIFAR-10. "
                "Upload an image to classify, reconstruct, or extract features.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Classify an uploaded image using MAE encoder + linear probe.

    Returns top-5 CIFAR-10 class probabilities.
    """
    file_bytes = await file.read()
    image_tensor = load_image(file_bytes)

    features = encode(image_tensor)
    pooled = features.mean(dim=1)  # (1, EMBED_DIM)

    with torch.no_grad():
        logits = classifier(pooled)

    probs = torch.softmax(logits, dim=1).squeeze()

    top5 = torch.topk(probs, k=5)

    predictions = [
        {"class": CIFAR10_CLASSES[idx], "confidence": round(conf.item(), 4)}
        for idx, conf in zip(top5.indices, top5.values)
    ]

    return {"predictions": predictions}


@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    """Upload an image → mask 75% of patches → reconstruct with MAE decoder.

    Returns base64-encoded PNGs of the original, masked, and reconstructed images.
    """
    file_bytes = await file.read()
    image_tensor = load_image(file_bytes)

    patches = patchify(image_tensor)

    visible_patches, mask, ids_keep, ids_restore = random_masking(patches)

    with torch.no_grad():
        pred = mae(visible_patches, ids_keep, ids_restore)

    # Build masked image (zero out masked patches)
    mask_expanded = mask.unsqueeze(-1).repeat(1, 1, PATCH_DIM)  # (1, 64, 48)
    masked_patches = patches * (1 - mask_expanded)
    masked_image = unpatchify(masked_patches).squeeze(0)  # (C, H, W)

    # Reconstructed: use predicted patches for masked positions, original for visible
    reconstructed_patches = patches * (1 - mask_expanded) + pred * mask_expanded
    reconstructed_image = unpatchify(reconstructed_patches).squeeze(0)

    original_image = image_tensor.squeeze(0)

    return {
        "original": tensor_to_base64_png(original_image),
        "masked": tensor_to_base64_png(masked_image),
        "reconstructed": tensor_to_base64_png(reconstructed_image),
    }


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    """Extract the 192-dim MAE encoder feature vector for an uploaded image.

    Features are computed by global average pooling over all patch embeddings.
    """
    file_bytes = await file.read()
    image_tensor = load_image(file_bytes)

    features = encode(image_tensor)
    pooled = features.mean(dim=1).squeeze()  # (EMBED_DIM,)

    return {"embedding": pooled.tolist(), "dim": EMBED_DIM}
