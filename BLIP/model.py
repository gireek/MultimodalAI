import timm
import torch
import torch.nn as nn



class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim=384, decoder_dim=384, num_decoder_layers=8, n_heads=8, ff_dim=1024, dropout=0.1):
        super(ImageCaptioningModel, self).__init__()

        # small vision vit
        self.encoder = timm.create_model('vit_small_patch16_224', pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False  # freeze

        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        # pytorch inbuilt transformerdecoder used for cross attention in forward function
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(decoder_dim, vocab_size)


    def forward(self, images, captions, device):
        """
        images: batch of images as tensors [batch, 3, 224, 224]
        captions: batch of caption token indices [batch, seq_len]
        Returns: logits over vocabulary for each time step [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = captions.size(0), captions.size(1)
        image_features = self.encoder.forward_features(images)   # torch.Size([batch_size, 197, 384])
        # print(image_features.shape)

        caption_embeddings = self.token_embedding(captions).to(device)  # [batch, seq_len, embed_dim]
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)  # shape [1, seq_len]
        pos_embeds = self.position_embedding(positions)  # [1, seq_len, embed_dim]
        caption_embeddings = caption_embeddings + pos_embeds  # [batch, seq_len, embed_dim]
        # the causal mask torch function
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device) #shape: [seq_len, seq_len]

        # memory is image_features and target is caption_embeddings
        # that means key and value matrice will come from image_features
        decoder_out = self.decoder(tgt=caption_embeddings, memory=image_features, tgt_mask=tgt_mask)
        outputs = self.output_proj(decoder_out)  # [batch, seq_len, vocab_size]
        return outputs
