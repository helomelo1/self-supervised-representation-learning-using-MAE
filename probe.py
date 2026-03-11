import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_dataloaders, patchify, random_masking
from model import MAE

# Hyperparameters
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

PATCH_SIZE = 4
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3
NUM_PATCHES = (32 // PATCH_SIZE) ** 2

EMBED_DIM = 192

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3

# Data
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# Model
mae = MAE(patch_dim=PATCH_DIM, num_patches=NUM_PATCHES).to(DEVICE)
mae.load_state_dict(torch.load("weights/mae_cifar10_150ep.pth", map_location=DEVICE))

encoder = mae.encoder

for p in encoder.parameters():
    p.requires_grad = False

encoder.eval()

# Classifier
classifier = nn.Linear(EMBED_DIM, 10).to(DEVICE)
optimizer = optim.Adam(classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(EPOCHS):
    classifier.train()
    total_loss = 0.0

    progress = tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}")

    for images, labels in progress:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        patches = patchify(images).to(DEVICE)

        B = patches.shape[0]

        ids_keep = torch.arange(NUM_PATCHES).unsqueeze(0).repeat(B, 1).to(DEVICE)

        with torch.no_grad():
            features = encoder(patches, ids_keep)

        # global avg. pooling
        features = features.mean(dim=1)
        logits = classifier(features)
        
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# Eval
classifier.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in tqdm(test_loader, desc="Testing"):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        patches = patchify(images).to(DEVICE)

        B = patches.shape[0]

        ids_keep = torch.arange(NUM_PATCHES).unsqueeze(0).repeat(B, 1).to(DEVICE)

        features = encoder(patches, ids_keep)
        features = features.mean(dim=1)

        logits = classifier(features)

        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"\nLinear Probe Accuracy: {accuracy:.2f}%")
