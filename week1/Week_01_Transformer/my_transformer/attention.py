import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        #todo
        # Attention(Q, K, V) = softmax(Q(K^T) / sqrt(dim_k))V
        
        # Q(K^T)
        qk = torch.matmul(q,  k.transpose(-2, -1))
        
        # Q(K^T) / sqrt(dim_k), scaling
        score = qk/math.sqrt(k.size(-1))
 
        # masking
        if mask is not None:
            score.masked_fill_(mask==0, -1e9)
        
        # softmax(Q(K^T) / sqrt(dim_k))
        score = F.softmax(score, dim=1)
        
        # [softmax(Q(K^T) / sqrt(dim_k))] V
        ouput = torch.matmul(score, v)
        
        return (ouput, score)
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #todo
        batch_size = Q.size(0)
        
        def split(x, fc):
            """
                input: (batch_size * seq_len * dim_embed)
                output: (batch_size * seq_len * n_heads * dim_k),   dim_model -> (n_heads * dim_k)
            """
            out = fc(x).view(batch_size, -1, self.n_heads, self.d_model//self.n_heads).transpose(1, 2)
            return out
        
        q = split(Q, self.query_layers)
        k = split(K, self.key_layers)
        v = split(V, self.value_layers)
        
        output = self.attention(q, k, v, mask)[0]
        output = output.transpose(1, 2).contiguous() # (batch_size, seq_len_q, n_heads, dim_k)
        output = output.view(batch_size, -1, self.n_heads * self.d_model) # (batch_size, seq_len_q, d_model * n_heads)
        
        output = self.fc(output) #(batch_size, seq_len_q, dim_embed)
        return output
        
        