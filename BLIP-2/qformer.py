import torch
from torch import nn
import torch.nn.functional as F

class QFormerBlock(nn.Module):
    """
    A single Transformer-like block with self attention + cross with imgs
    """
    def __init__(self, d_model=384, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # Self-Attention(queries + question tokens)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Cross-Attention(to image embeddings)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, image_embeds):
        """
        Args:
          x:            [B, (N + Q), d_model] (concatenation of query + question embeddings)
          image_embeds: [B, I, d_model]  (the set of image patch embeddings from a vision encoder)
        Returns: x: [B, (N + Q), d_model] (updated embeddings)
        """
        # Self-Attention among Queries + Question
        x_res = x
        x = self.norm1(x)
        x_attn, _ = self.self_attn(query=x, key=x, value=x)
        x = x_res + self.dropout(x_attn)

        # Cross-Attention step
        x_res = x
        x = self.norm2(x)
        x_attn, _ = self.cross_attn(query=x, key=image_embeds, value=image_embeds)
        x = x_res + self.dropout(x_attn)
        
        x_res = x
        x = self.norm3(x)
        x_ff = self.linear2(F.relu(self.linear1(x))) #feedforward
        x = x_res + self.dropout(x_ff)
        
        return x



class QFormer(nn.Module):
    """
    A minimal Q-Former example with:
      - Learnable Query Embeddings of size N
      - An embedding layer for question tokens
      - A stack of QFormerBlocks that do self-attn + cross-attn
      - We extract only the first N query embeddings from the final output
    """
    def __init__(self,vocab_size,max_qn_len,num_queries=32,d_model=384,nhead=4,num_blocks=3,dim_feedforward=1024):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        
        # Learnable query embeddings. shape: [num_queries, d_model]
        self.learned_queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding =  nn.Embedding(max_qn_len, d_model)
        self.blocks = nn.ModuleList([
            QFormerBlock(d_model, nhead, dim_feedforward) for _ in range(num_blocks)
        ])

    def forward(self, question_ids, image_embeds):
        """
        Args:
          question_ids: [B, Q] (token IDs for the question)
          image_embeds: [B, I, d_model] (e.g., patch embeddings from vision encoder)
        Returns:
          query_outputs: [B, N, d_model], the final query embeddings that represent 
                         the image (conditioned on the question).
        """
        B = question_ids.size(0)
        question_ids = question_ids.to(image_embeds.device)
        
        #Embed question tokens [B, Q, d_model]
        question_embs = self.text_embedding(question_ids)
        question_embs += self.pos_embedding(torch.arange(question_ids.size(1), device=image_embeds.device))
        
        #Learned query embeddings [B, N, d_model]
        query_embs = self.learned_queries.unsqueeze(0).expand(B, -1, -1)
        
        #Concatenate: [Queries + Question] => shape [B, (N + Q), d_model]
        x = torch.cat([query_embs, question_embs], dim=1)
        for block in self.blocks:
            x = block(x, image_embeds)
        
        # Extract the first N tokens (the queries) from the final output [B, N, d_model]
        query_outputs = x[:, :self.num_queries, :]
        return query_outputs