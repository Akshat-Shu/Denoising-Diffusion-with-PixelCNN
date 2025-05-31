import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim, embedding_dim, timesteps=1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.timesteps = timesteps
        self.time_dim = time_dim

        self.half_dim = embedding_dim // 2
        embedding = torch.arange(0, self.time_dim, step=2, dtype=torch.float32) / self.time_dim
        embedding = torch.exp(- embedding * torch.log(10000))
        time_embedding = torch.arange(0, timesteps).float()
        time_embedding = time_embedding.unsqueeze(1) * embedding.unsqueeze(0)
        time_embedding = torch.cat((time_embedding.sin(), time_embedding.cos()), dim=1)
        time_embedding = time_embedding.view(timesteps, self.half_dim * 2)

        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(time_embedding, freeze=True),
            nn.Linear(self.half_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        nn.init.xavier_uniform_(self.embedding[1].weight)
        nn.init.xavier_uniform_(self.embedding[3].weight)
        self.embedding[1].bias.data.fill_(0)
        self.embedding[3].bias.data.fill_(0)

    def forward(self, t):
        return self.embedding(t)