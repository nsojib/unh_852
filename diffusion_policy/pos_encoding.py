import torch
import torch.nn as nn
import math 

# my implementation
class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        
        self.d_model = d_model
        self.max_len = max_len  
        t = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # positions 

        fs=[1.0 /( 10000**(2*i/d_model) ) for i in range(0, d_model//2)]  #high to low frequency
        fs= torch.tensor(fs)
    
        sins=torch.sin(t * fs)
        coss=torch.cos(t * fs) 
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = sins
        pe[:, 1::2] = coss  
        self.register_buffer('pe', pe.unsqueeze(0)) 
        
    def forward(self, timesteps): 
        X= timesteps 
        X = self.pe[:, :X.size(-1)]   
        return X.squeeze(0)
    

# diffusion policy implementation
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb