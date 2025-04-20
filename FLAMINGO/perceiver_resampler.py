import torch
from torch import nn


class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_layers, num_latents, heads):
        # Consider here dimension dim for both input patches and latents
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # Positional embeddings for the latents
        # I dont add position embeddings to the image patches considering
        # they already come with position info from a ViT like model 
        self.latent_pos_emb = nn.Parameter(torch.randn(num_latents, dim))

        # A single TransformerDecoder layer for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = dim,
            nhead = heads, #num attention heads
            dim_feedforward = int(dim * 4),
            activation = "gelu",
            batch_first = True  
        )
        # Stack multiple TransformerDecoder layers
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        b, n, d = x.shape # n is number of image patches
        # Expand latents from [num_latents, dim] to [batch, num_latents, dim]
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        latents = latents + self.latent_pos_emb.unsqueeze(0)

        # Use the latents as "target" (tgt), and x as "memory"
        # since its the latent being conditioned with the image patches
        # so the K and V come from x but the Q matrice from latents
        latents = self.decoder(tgt=latents, memory=x) 
        # final shape- [batch, num_latents, dim]

        return self.norm(latents)