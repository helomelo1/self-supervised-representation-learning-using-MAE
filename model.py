import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4):
        super().__init__()

        hidden_dim = embed_dim * mlp_ratio

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
    

class MAEEncoder(nn.Module):
    def __init__(self, patch_dim, num_patches, embed_dim=192, depth=6, num_heads=3):
        super().__init__()

        self.patch_embed = nn.Linear(patch_dim, embed_dim) # Patch Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # Positional Embedding
        self.blocks = nn.ModuleList(
            [ViTBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visible_patches, ids_keep):
        x = self.patch_embed(visible_patches)

        pos = torch.gather(
            self.pos_embed.repeat(x.shape[0], 1, 1),
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        x = x + pos

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x
    

class MAEDecoder(nn.Module):
    def __init__(self, num_patches, patch_dim=48, embed_dim=192, decoder_embed_dim=128, depth=6, num_heads=4):
        super().__init__()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim) # encoder output projection to decoder dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) # mask tokens
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim)) # positional encoding

        self.decoder_blocks = nn.ModuleList(
            [ViTBlock(decoder_embed_dim, num_heads) for _ in range(depth)]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim)

    def forward(self, latent_tokens, ids_restore):
        B, N_vis, _ = latent_tokens.shape
        N = ids_restore.shape[1]

        x = self.decoder_embed(latent_tokens)
        mask_tokens = self.mask_token.repeat(B, N - N_vis, 1)

        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        return x
    

class MAE(nn.Module):
    def __init__(self, patch_dim, num_patches, embed_dim=192):
        super().__init__()

        self.encoder = MAEEncoder(
            patch_dim=patch_dim,
            num_patches=num_patches,
            embed_dim=embed_dim
        )

        self.decoder = MAEDecoder(
            num_patches=num_patches,
            embed_dim=embed_dim,
            patch_dim=patch_dim
        )

    def forward(self, visible_patches, ids_keep, ids_restore):
        latent = self.encoder(visible_patches, ids_keep)
        pred = self.decoder(latent, ids_restore)

        return pred