class blip_vqa(nn.Module):
  def __init__(self, vocab_size, num_queries=8,
               hidden_dim=384, nhead = 8,
               dim_feedforward=512, dropout = 0.15,
               num_layers = 3, num_decoder_layers=6,
               max_len=40, embed_dim=384 ):
    super(blip_vqa, self).__init__()

    # PART 1:
    # small vision vit
    self.encoder = timm.create_model('vit_small_patch16_224', pretrained=True)
    for param in self.encoder.parameters():
        param.requires_grad = False  # freeze


    # PART 2:
    # QFORMER
    self.qformer = Qformer(num_queries=num_queries, hidden_dim=hidden_dim, nhead=nhead,
                           dim_feedforward=dim_feedforward, dropout=dropout, num_layers=num_layers)

    # PART 3:
    self.token_embedding = nn.Embedding(vocab_size, embed_dim)
    self.position_embedding = nn.Embedding(max_length, embed_dim)

    decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
    self.output_proj = nn.Linear(hidden_dim, vocab_size)


  def forward(self, images, questions, answers, device):
    batch_size, seq_len = questions.size(0), questions.size(1)
    image_features = self.encoder.forward_features(images)   # torch.Size([batch_size, 197, 384])
    positions = torch.arange(0, seq_len, device=device).unsqueeze(0)  # shape [1, seq_len]
    pos_embeds = self.position_embedding(positions)  # [1, seq_len, embed_dim]

    question_embeds = self.token_embedding(questions).to(device)  # [batch, seq_len, embed_dim]
    question_embeds += pos_embeds  # [batch, seq_len, embed_dim]
    learned_queries = self.qformer(image_features, questions, question_embeds).to(device)          # torch.Size([batch_size, 8, 384])

    answer_embed = self.token_embedding(answers).to(device)  # [batch, seq_len, embed_dim]
    answer_embed = answer_embed + pos_embeds  # [batch, seq_len, embed_dim]

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device) #shape: [seq_len, seq_len]

    pad_token_id = 0  # assume <pad>=0
    tgt_key_padding_mask = (answers == pad_token_id)

    # memory is image_features and target is answer_embed
    decoder_out = self.decoder(tgt=answer_embed, memory=learned_queries, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    outputs = self.output_proj(decoder_out)  # [batch, seq_len, vocab_size]
    return outputs