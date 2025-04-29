import torch
from torch import nn
import torch.nn.functional as F
import timm
from qFormer import QFormer

class blip_vqa(nn.Module):

  def __init__(self, vocab_size, max_qn_len, max_ans_len, hidden_dim=384, nhead = 8,
               dim_feedforward=512, dropout = 0.15, num_decoder_layers=6):
    super(blip_vqa, self).__init__()

    # PART 1: small vision vit
    self.encoder = timm.create_model('vit_small_patch16_224', pretrained=True)
    for param in self.encoder.parameters():
        param.requires_grad = False  # freeze

    # PART 2: QFORMER
    self.qformer = QFormer(vocab_size, max_qn_len)

    # PART 3: setup for the language model
    self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
    self.position_embedding = nn.Embedding(max_ans_len, hidden_dim)

    decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, 
                    dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
    self.output_proj = nn.Linear(hidden_dim, vocab_size)




  def forward(self, images, questions, answers, data, device):
    _, seq_len = questions.size(0), questions.size(1)
    image_features = self.encoder.forward_features(images)   # torch.Size([batch_size, 197, 384])
    learned_queries = self.qformer(questions, image_features).to(device)          # torch.Size([batch_size, 8, 384])

    pos_embeds = self.position_embedding(torch.arange(answers.size(1), device=device))
    answer_embed = self.token_embedding(answers).to(device)  # [batch, seq_len, embed_dim]
    answer_embed = answer_embed + pos_embeds  # [batch, seq_len, embed_dim]

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(answers.size(1)).to(device) #shape: [seq_len, seq_len]
    # check for pad token so that attnetion doesnt learn about this
    tgt_key_padding_mask = (answers == 0)

    # memory is image_features and target is answer_embed
    decoder_out = self.decoder(tgt=answer_embed, memory=learned_queries, tgt_mask=tgt_mask, 
                               tgt_key_padding_mask=tgt_key_padding_mask)
    outputs = self.output_proj(decoder_out)  # [batch, seq_len, vocab_size]
    return outputs




  def generate(self, images, questions, max_ans_len, device, data):
    image_features = self.encoder.forward_features(images.to(device))
    learned_queries = self.qformer(questions.to(device), image_features)

    num_questions = questions.size(0)
    # bcz 1 is <bos>
    generated = torch.tensor([1]).unsqueeze(0).repeat(num_questions, 1).to(device)
    curr_pos = torch.tensor([0]).unsqueeze(0).repeat(num_questions, 1).to(device)

    for i in range(max_ans_len):
      pos_embeds = self.position_embedding(curr_pos)
      answer_embed = self.token_embedding(generated)
      answer_embed = answer_embed + pos_embeds

      tgt_mask = nn.Transformer.generate_square_subsequent_mask(curr_pos.size(1)).to(device)
      decoded = self.decoder(tgt=answer_embed, memory=learned_queries, tgt_mask=tgt_mask)
      outputs = self.output_proj(decoded)

      new_pos = torch.tensor([i + 1]).unsqueeze(0).repeat(num_questions, 1).to(device)
      curr_pos = torch.cat([curr_pos, new_pos], dim=1)
      output_last = outputs[:, -1, :]
      generated = torch.cat([generated, output_last.argmax(dim=1).unsqueeze(1)], dim=1)

    sentences = []
    for gen in generated:
      sentence = []
      for i in range(gen.size(0)):
        if data.rev_vocab[gen[i].item()] == "<eos>":
          break
        else:
          sentence.append(data.rev_vocab[gen[i].item()])
      sentences.append(" ".join(sentence))

    return sentences