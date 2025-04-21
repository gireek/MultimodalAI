import torch
from torch import nn

def compare_tensors_batch(text_tokens, media_tokens, only_preceding_media):
    """
    text_tokens:  [b, x]
    media_tokens: [b, y]
    returns a bool mask [b, x, y] where True means "masked out"
    """
    b, x= text_tokens.shape
    _, y= media_tokens.shape

    text_expanded = text_tokens.unsqueeze(2).expand(b, x, y)
    media_expanded = media_tokens.unsqueeze(1).expand(b, x, y)

    if only_preceding_media:
        comparison = (text_expanded == media_expanded)
    else:
        comparison = (text_expanded >= media_expanded)

    # We invert comparison so True => "mask out"
    mask_out = ~comparison
    return mask_out



class MaskedCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, only_attend_immediate_media=True):
        super().__init__()
        self.only_attend_immediate_media = only_attend_immediate_media
        self.norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, text_tokens, media, media_locations=None):
        """
        text_tokens: (b, n, dim) - text tokens
        media: (b, t, m, dim) - media features (e.g., image tokens over time)
        media_locations: (b, n) - bool, True if token follows a media frame
        """
        b, t, m, d = media.shape  # t is number of images and m is the latent dim
        device = text_tokens.device

        text_tokens= self.norm(text_tokens)  # normalize text tokens
        media = media.view(b, t * m, d)  # flatten

        # if media locations for batch 1 is [false, false, true, false, false, true, false, true]
        # cumsum would be [0, 0, 1, 1, 1, 2, 2, 3] n
        # media batch 1 is [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] so basically m is 4 and t is 3
        text_cumsum = torch.cumsum(media_locations, dim = -1) # shape n
        media_nums = torch.arange(1, t+1).repeat_interleave(m)
        media_nums = media_nums.unsqueeze(0).repeat(b, 1)
        
        attn_mask = compare_tensors_batch(text_cumsum, media_nums, self.only_attend_immediate_media)
        output, _ = self.cross_attn(query=text_tokens,
                                    key=media,
                                    value=media,
                                    attn_mask=attn_mask
                                )

        return output