import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, out_channels, cond_dim):
        super().__init__()
        # FiLM modulation https://arxiv.org/abs/1709.07871 
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

    def forward(self, x, cond):
        embed = self.cond_encoder(cond)

        embed = embed.reshape( embed.shape[0], 2, self.out_channels, 1)
        gamma = embed[:,0,...]
        beta = embed[:,1,...]
        
        out = gamma * x + beta    # FiLM modulation
        return out
    