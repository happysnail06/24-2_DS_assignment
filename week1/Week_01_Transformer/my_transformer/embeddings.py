import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        #todo
        encoding = torch.zeros(max_len, d_model)
        encoding.requires_grad = False
        
        # Position 생성, 범위: 0 ~ (max_len - 1)
        # 각 단어의 위치 정보를 기록하는 역할
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        
        """
            -(math.log(10000.0) / d_model)): 스케일링 팩터
            torch.arange(0, d_model, 2): 짝수 인덱스 생성
            exp: 각 차원의 frequency를 다르게 하기 위함
            
            intuition: 각 위치에 유니크한 패턴을 부여
        """
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        # 짝수 index에는 sin함수를, 홀수 index에는 cos함수를 사용, 
        # intuition: 결국 시퀀스 순서를 이해하고, 문맥 내에서 단어 간의 관계를 더 잘 학습할 수 있게 하는 것
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.encoding = encoding.unsqueeze(0)
    
    def forward(self, x: Tensor) -> Tensor:
        #todo one line!
        return self.encoding[:, :x.size(1), :]