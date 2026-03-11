import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_dataloaders, patchify
from model import MAE

# device
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

PATCH_SIZE = 4
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3
NUM_PATCHES = (32 // PATCH_SIZE) ** 2

BATCH_SIZE = 128

# odd k values
K_VALUES = list(range(1, 50, 2))  # 1,3,5,...,21

# Data
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# Model
mae = MAE(patch_dim=PATCH_DIM, num_patches=NUM_PATCHES).to(DEVICE)
mae.load_state_dict(torch.load("weights/mae_cifar10_150ep.pth", map_location=DEVICE))

encoder = mae.encoder
encoder.eval()

# Build Feature Bank
feature_bank = []
label_bank = []

with torch.no_grad():

    for images, labels in tqdm(train_loader, desc="Building feature bank"):

        images = images.to(DEVICE)
        patches = patchify(images).to(DEVICE)

        B = patches.shape[0]
        ids_keep = torch.arange(NUM_PATCHES).unsqueeze(0).repeat(B, 1).to(DEVICE)

        features = encoder(patches, ids_keep)
        features = features.mean(dim=1)

        features = F.normalize(features, dim=1)

        feature_bank.append(features.cpu())
        label_bank.append(labels)

feature_bank = torch.cat(feature_bank)
label_bank = torch.cat(label_bank)

# kNN Evaluation
correct = {k: 0 for k in K_VALUES}
total = 0

max_k = max(K_VALUES)

with torch.no_grad():

    for images, labels in tqdm(test_loader, desc="kNN evaluation"):

        images = images.to(DEVICE)
        patches = patchify(images).to(DEVICE)

        B = patches.shape[0]
        ids_keep = torch.arange(NUM_PATCHES).unsqueeze(0).repeat(B, 1).to(DEVICE)

        features = encoder(patches, ids_keep)
        features = features.mean(dim=1)

        features = F.normalize(features, dim=1)

        similarity = torch.mm(features.cpu(), feature_bank.T)

        topk = similarity.topk(k=max_k, dim=1).indices
        neighbors = label_bank[topk]

        for k in K_VALUES:

            preds = torch.mode(neighbors[:, :k], dim=1).values
            correct[k] += (preds == labels).sum().item()

        total += labels.size(0)

# Print Results]
print("\nkNN Accuracy Results")
print("---------------------")

best_acc = 0
best_k = None

for k in K_VALUES:

    acc = 100 * correct[k] / total
    print(f"k = {k:<2} → {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_k = k

print("\nBest kNN accuracy")
print(f"k = {best_k}, accuracy = {best_acc:.2f}%")