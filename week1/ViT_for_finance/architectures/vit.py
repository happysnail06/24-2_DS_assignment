import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from timm.models.layers import trunc_normal_

"""
"vision_transformer": {
        "learning_rate_type": "Not found",  # 0.001
        "weight_decay": 0.0001,
        "batch_size": 128,
        "num_epochs": 200,
        "image_size": 65,
        "patch_size": 8,  # Size of the patches to be extract from the input images
        "projection_dim": 64,  # 128
        "num_heads": 4,
        "transformer_layers": 8,  # 6, 8, 10
        "mlp_head_units": [2048, 1024],
        "num_classes": 3,
    }
    https://csm-kr.tistory.com/54
"""
class EmbeddingLayer(nn.Module):
    def __init__(self, img_size: int = 65, patch_size: int = 8, in_channels: int = 1, embed_dim: int = 64)-> None:
        super(EmbeddingLayer, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # class
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        
        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02) # not sure what this is
        

    def forward(self, x) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)  # (batch_size, embed_dim, num_patches_h, num_patches_w) -> (batch_size, num_patches, embed_dim)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        
        x = torch.cat((cls_tokens, x), dim=1)  #(batch_size, num_patches + 1, embed_dim)
        
        x = x + self.pos_embed  #(batch_size, num_patches + 1, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1, qkv_bias=False) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, embed_dim = x.size()
        
        # Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k, v = qkv.unbind(0)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_length, seq_length)
        probs = F.softmax(scores, dim=-1)  #(batch_size, num_heads, seq_length, seq_length)
        probs = self.attn_dropout(probs)
        
        output = torch.matmul(probs, v).transpose(1, 2).reshape(batch_size, seq_length, embed_dim)  # .reshape(B, N, C), batch_size, num_heads, seq_length, head_dim) -> (batch_size, seq_length, embed_dim)
        
        # Final projection
        output = self.proj(output)  #(batch_size, seq_length, embed_dim)
        output = self.proj_dropout(output)
        
        return output

class MLP(nn.Module):
    def __init__(self, embed_dim=64, mlp_dim=256, dropout=0.1, bias=True) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim, bias=bias)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(mlp_dim, embed_dim, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, mlp_dim=256, dropout=0.1, qkv_bias=False) -> None:
        super(Block, self).__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, qkv_bias=qkv_bias, dropout=dropout)
        self.LayerNorm1 = nn.LayerNorm(embed_dim)
    
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.LayerNorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x) -> torch.Tensor:
        x = x + self.attn(self.LayerNorm1(x))
        x = x + self.mlp(self.LayerNorm2(x))
        return x

    #여기까지 Encoder 구현 끝!!
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size=65, patch_size=8, in_channels=1, num_classes=3, 
                 embed_dim=64, num_heads=4, num_layers=8, qkv_bias=False, mlp_dim=256, dropout=0.1) -> None:
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        self.embedding = EmbeddingLayer(img_size, patch_size, in_channels, embed_dim)
        
        self.blocks = nn.Sequential(*[
            Block(
                embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, qkv_bias=qkv_bias, dropout=dropout)
            for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.embedding(x)  #(batch_size, num_patches + 1, embed_dim)
        
        x = self.blocks(x)
        
        x = self.norm(x)
        
        x = x[:, 0]  # Extract the class token
        x = self.head(x)
        
        return x
