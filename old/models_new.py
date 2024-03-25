import torch
import torch.nn as nn
import numpy as np
import math

# Inspired by MDTv2
# https://github.com/sail-sg/MDT

# Sources
# https://github.com/sail-sg/MDT/blob/main/masked_diffusion/models.py
# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        assert frequency_embedding_size % 2 == 0, "Frequency embedding size must be even."
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        half = frequency_embedding_size // 2
        self.freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half)[None]

    def forward(self, t):
        # Sinusoidal timestep embeddings
        args = t[:, None].float() * torch.tensor(self.freqs, device=t.device)
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        # MLP
        t_emb = self.mlp(embedding)
        return t_emb
    

