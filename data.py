import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR10 normalization values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
PATCH_SIZE = 4


def get_transforms(img_size=32, train=True):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transform


def random_masking(patches, mask_ratio=0.75):
    """
    Mask random 75% of the patches of an image

    Args:
        patches: One Image divided into N patches
        mask_ratio: deafult: 0.75

    Returns:
        visible_patches
        mask
        ids_restore
    """
    B = patches.shape[0]
    N = patches.shape[1]
    patch_dim = patches.shape[2]

    noise = torch.rand(B, N, device=patches.device)
    ids_shuffle = torch.argsort(noise, dim=1)

    len_keep = int(N * (1 - mask_ratio))
    ids_keep = ids_shuffle[:, :len_keep]

    visible_patches = torch.gather(patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, patch_dim))
    mask = torch.ones(B, N, device=patches.device)
    mask[:, :len_keep] = 0

    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return visible_patches, mask, ids_restore


def patchify(images, patch_size=PATCH_SIZE):
    """
    Convert images into patches for MAE.
    
    Args:
        images: Tensor of shape (B, C, H, W)
        patch_size: Size of each patch (default 4 for CIFAR10 32x32 -> 8x8 patches)
    
    Returns:
        patches: Tensor of shape (B, num_patches, patch_size * patch_size * C)
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image dimensions ({H}, {W}) must be divisible by patch_size ({patch_size})"
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Reshape: (B, C, H, W) -> (B, C, n_h, p_h, n_w, p_w)
    patches = images.reshape(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    # Permute: (B, C, n_h, p_h, n_w, p_w) -> (B, n_h, n_w, p_h, p_w, C)
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    # Flatten patches: (B, n_h, n_w, p_h, p_w, C) -> (B, num_patches, patch_size^2 * C)
    patches = patches.reshape(B, num_patches, -1)
    
    return patches


def unpatchify(patches, patch_size=PATCH_SIZE, channels=3):
    """
    Reconstruct images from patches.
    
    Args:
        patches: Tensor of shape (B, num_patches, patch_size * patch_size * C)
        patch_size: Size of each patch
        channels: Number of image channels
    
    Returns:
        images: Tensor of shape (B, C, H, W)
    """
    B, num_patches, _ = patches.shape
    num_patches_side = int(num_patches ** 0.5)
    H = W = num_patches_side * patch_size
    
    # Reshape: (B, num_patches, patch_dim) -> (B, n_h, n_w, p_h, p_w, C)
    patches = patches.reshape(B, num_patches_side, num_patches_side, patch_size, patch_size, channels)
    # Permute: (B, n_h, n_w, p_h, p_w, C) -> (B, C, n_h, p_h, n_w, p_w)
    patches = patches.permute(0, 5, 1, 3, 2, 4)
    # Reshape: (B, C, n_h, p_h, n_w, p_w) -> (B, C, H, W)
    images = patches.reshape(B, channels, H, W)
    
    return images


def get_dataloaders(batch_size=128, num_workers=0, data_dir='./data'):
    """
    Get CIFAR10 train and test dataloaders.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        data_dir: Directory to store/load CIFAR10 data
    
    Returns:
        train_loader, test_loader
    """
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

# SANITY CHECK!
if __name__ == "__main__":
    train_loader, _ = get_dataloaders()

    images, _ = next(iter(train_loader))

    patches = patchify(images)

    visible, mask, ids_restore = random_masking(patches)

    print(patches.shape)   # (B, 64, 48)
    print(visible.shape)   # (B, 16, 48)
    print(mask.shape)      # (B, 64)
    print(ids_restore.shape)