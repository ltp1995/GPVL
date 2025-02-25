import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention
        residual = x
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = self.norm1(x + residual)
        
        # Feedforward
        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.norm2(x + residual)
        
        return x

class TransformerEncoder_ltp(nn.Module):
    def __init__(self, num_layers, dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoder_ltp, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x