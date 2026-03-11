import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE

from data import get_dataloaders, patchify
from model import MAE

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

PATCH_SIZE = 4
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3
NUM_PATCHES = (32 // PATCH_SIZE) ** 2

# load data
_, test_loader = get_dataloaders(batch_size=128)

# load model
mae = MAE(patch_dim=PATCH_DIM, num_patches=NUM_PATCHES).to(DEVICE)
mae.load_state_dict(torch.load("weights/mae_cifar10_150ep.pth", map_location=DEVICE))

encoder = mae.encoder
encoder.eval()

features_list = []
labels_list = []

# collect features
with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(DEVICE)
        patches = patchify(images).to(DEVICE)

        B = patches.shape[0]
        ids_keep = torch.arange(NUM_PATCHES).unsqueeze(0).repeat(B,1).to(DEVICE)

        features = encoder(patches, ids_keep)
        features = features.mean(dim=1)

        features_list.append(features.cpu())
        labels_list.append(labels)

features = torch.cat(features_list)
labels = torch.cat(labels_list)

# reduce dimensionality
tsne = TSNE(n_components=2, perplexity=30)
embedding = tsne.fit_transform(features)

# plot
plt.figure(figsize=(8,8))
scatter = plt.scatter(
    embedding[:,0],
    embedding[:,1],
    c=labels,
    cmap="tab10",
    s=10
)

plt.colorbar(scatter)
plt.title("t-SNE of MAE Representations (CIFAR-10)")
plt.show()