import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_dataloaders, patchify, random_masking
from model import MAE

# Hyperparameters
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PATCH_SIZE = 4
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3
NUM_PATCHES = (32 // PATCH_SIZE) ** 2

BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-4

# Data
train_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

# Model
model = MAE(
    patch_dim=PATCH_DIM,
    num_patches=NUM_PATCHES,
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):

    total_loss = 0.0

    # tqdm progress bar for batches
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for images, _ in progress_bar:

        images = images.to(DEVICE)

        # patchify
        patches = patchify(images)

        # masking
        visible_patches, mask, ids_keep, ids_restore = random_masking(patches)

        visible_patches = visible_patches.to(DEVICE)
        mask = mask.to(DEVICE)
        ids_keep = ids_keep.to(DEVICE)
        ids_restore = ids_restore.to(DEVICE)

        # forward pass
        pred = model(visible_patches, ids_keep, ids_restore)

        # reconstruction loss
        loss = (pred - patches) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # update tqdm display
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "mae_cifar10.pth")