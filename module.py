import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class MultiHeadAttention(nn.Module): 
    
    def __init__(self, embed_dim = 2048, n_heads = 8, drop=0.1): 
        
        super().__init__()
        
        assert embed_dim % n_heads == 0, 'the dimension of input embedding vector must be divisble by the number of attention head'
        
        self.embed_dim = embed_dim 
        self.head_num = n_heads 
        self.scale = (embed_dim // n_heads) ** -0.5 
        
        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=True)
        
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x): 
        
        """
        Args: 
            x: the input vector in [batch_size, num_frames, seq_length, dim] or
                                   [batch_size, num_frames, dim]
        """
        if x.dim() == 4: 
            B, T, N, D = x.shape 
            qkv = self.qkv(x).reshape(B, T, 3, N, self.head_num, D // self.head_num).permute(2, 0, 1, 4, 3, 5)
            # 4 x 16 x 197 x 768 -> 4 x 16 x 197 x (768*3) -> 4 x 16 x 3 x 197 x 12 x 64 -> 3 x 4 x 16 x 12 x 197 x 64
        elif x.dim() == 3:
            B, T, D = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.head_num, D // self.head_num).permute(2, 0, 3, 1, 4)
            # 4 x 17 x 768 -> 4 x 17 x (768*3) -> 4 x 17 x 3 x 12 x 64 -> 3 x 4 x 12 x 17 x 64 

        q, k, v = qkv[0], qkv[1], qkv[2]   # 4 x 16 x 12 x 197 x 64 or 4 x 12 x 17 x 64
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale   #  4 x 16 x 12 x 197 x 197 or 4 x 12 x 17 x 17 
        
        scores = F.softmax(attn, dim = -1)   # 4 x 16 x 12 x 197 x 197 or 4 x 12 x 17 x 17 
        
        if x.dim == 4:
            x = torch.matmul(scores, v).permute(0, 1, 3, 2, 4).reshape(B, T, N, -1)
            # 4 x 16 x 12 x 197 x 64 -> 4 x 16 x 197 x 12 x 64 -> 4 x 16 x 197 x 768
            
        elif x.dim == 3:
            x = torch.matmul(scores, v).permute(0, 2, 1, 3).reshape(B, T, -1)
            # 4 x 12 x 17 x 64 -> 4 x 17 x 12 x 64
        
        x = self.proj(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    
    def __init__(self,embed_dim = 2048, n_heads = 8, expansion_factor = 4, drop=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim 
        self.head_num = n_heads
        self.expansion_factor = expansion_factor
        
        self.attention = MultiHeadAttention(embed_dim, n_heads, drop)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor), 
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1))
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        
    def forward(self, x): 
        """
        Args: 
            x: input vector [4 x 16 x 197 x 762]
        """
        
        x = self.attention(self.norm1(x)) + x 
        
        x = self.mlp(self.norm2(x)) + x
        
        return x

class Transformer(nn.Module): 
    
    def __init__(self, embed_dim=2048, n_heads=8, expansion_factor=4, L=4, drop=0.1): 
        super().__init__()
        
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, n_heads, expansion_factor, drop) for _ in range(L)])
        
    def forward(self, x):
        """
        Args: 
            x: input video frames 
            4 x 16 x 197 x 192
        """
     
        for layer in self.layers: 
            x = layer(x)
            
        return x

class AMTransformer(nn.Module): 

    def __init__(self, embed_dim=2048, n_heads=8, expansion_factor=4, L=4, drop=0.1, seq_length=300): 

        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))
        self.transformer = Transformer(embed_dim, n_heads, expansion_factor, L, drop)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.ReLU()
        )

    def forward(self, x): 

        """
        Arg:
            x: input layer-wise embeeding [batch_size, seq_length, embed_dim]
        """

        B, N, D = x.shape 
        print(x.shape)

        cls_token = self.cls_token.repeat(B, 1, 1)
        
        x = torch.cat((cls_token, x), dim=1)   # B x N x D ==> B x (N+1) X D

        x += self.pos_embedding[:, :(N+1), :]

        x = self.transformer(x)   # B x N+1 x D

        x = self.mlp(x[:,0,:]) 

        return x


