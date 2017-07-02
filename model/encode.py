
import torch
from torch import nn
from .attention import MultiHeadAttention
from .norm import LayerNorm

class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, output_dim, num_heads=8, dropout=0.0):
        super(Encoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.self_attention = MultiHeadAttention(self.embedding_dim,
                                                       self.output_dim,
                                                       self.num_heads,
                                                       self.dropout)
        self.layer_norm = LayerNorm()

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        residual = embedding + self.dropout(self.self_attention(embedding, embedding, embedding))
        return self.layer_norm(residual)


class Decoder(nn.Module):
    def __init__(self, num_embedding, embedding_dim, output_dim, num_heads=8, dropout=0.0):
        super(Decoder, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(self.num_embedding, self.embedding_dim)

        self.decoder_encoder_attention = MultiHeadAttention(self.embedding_dim,
                                                       self.output_dim,
                                                       self.num_heads,
                                                       self.dropout)

    def forward(self, input, output):
        embedding = self.embedding(inputs)
        residual = embedding + self.dropout(self.self_attention(embedding, embedding, embedding))

        return self.layer_norm(residual)