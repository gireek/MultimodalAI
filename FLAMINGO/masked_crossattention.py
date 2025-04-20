import torch
from torch import nn

def compare_tensors(text_tokens, media_tokens, only_preceding_media):
    # both are 1d tensors
    # the former with length n and latter with length t

    n, t = text_tokens.size(0), media_tokens.size(0)

    text_expanded = text_tokens.unsqueeze(1).expand(n, t)
    media_expanded = media_tokens.unsqueeze(0).expand(n, t)

    if only_preceding_media:
        comparison = text_expanded == media_expanded
    else:
        comparison = text_expanded >= media_expanded

    return ~comparison


class MaskedCrossAttention(nn.Module):
    def __init__(self, *, dim, heads=8, only_attend_immediate_media=True):
        super().__init__()
        self.heads = heads
        self.only_attend_immediate_media = only_attend_immediate_media

        self.norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
            bias=False
        )


    def forward(self, x, media, media_locations=None):
        """
        x: (b, n, dim) - text tokens
        media: (b, t, m, dim) - media features (e.g., image tokens over time)
        media_locations: (b, n) - bool, True if token follows a media frame
        """
        # gonna pass 1 batch at a time
        _, t, m, d = media.shape  # t is number of images and m is the latent dim
        device = x.device

        x = self.norm(x)  # normalize text tokens
        media = media.view(b, t * m, d)  # flatten bcz it has to be same shape as x, b is 1 btw

        if self.only_attend_immediate_media = False:
          out = self.cross_attn(query = x, key = media, value = media)
          return out

        # if media locations for batch 1 is [false, false, true, false, false, true, false, true]
        # cumsum would be [0, 0, 1, 1, 1, 2, 2, 3] n
        # media batch 1 is [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] so basically m is 4 and t is 3
        text_cumsum = torch.cumsum(media_location, dim = -1)[0] # shape n
        media_nums = torch.arange(1, t+1).repeat_interleave(m)
        print(media_nums.shape)
        print(text_cumsum.shape)