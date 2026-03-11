import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_dataloaders

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3

# CIFAR10 image size
INPUT_DIM = 32 * 32 * 3

# data
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# simple linear classifier
model = nn.Linear(INPUT_DIM, 10).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# training
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # flatten pixels
        images = images.view(images.size(0), -1)

        logits = model(images)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")


# evaluation
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in tqdm(test_loader, desc="Testing"):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        images = images.view(images.size(0), -1)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"\nPixel Baseline Accuracy: {accuracy:.2f}%")