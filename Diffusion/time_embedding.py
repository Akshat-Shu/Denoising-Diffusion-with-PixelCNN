import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim, embedding_dim, timesteps=1000, device='cpu'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.timesteps = timesteps
        self.time_dim = time_dim

        self.half_dim = self.time_dim // 2
        embedding = torch.arange(0, self.time_dim, step=2, dtype=torch.float32) / self.time_dim
        embedding = torch.exp(- embedding * torch.log(torch.tensor(10000)))
        time_embedding = torch.arange(0, timesteps).float()
        time_embedding = time_embedding.unsqueeze(1) * embedding.unsqueeze(0)
        # print(f"Time embedding shape before sin/cos: {time_embedding.shape}, expected: ({timesteps}, {self.half_dim})")
        time_embedding = torch.cat((time_embedding.sin(), time_embedding.cos()), dim=1)
        # print(f"Time embedding shape: {time_embedding.shape}, expected: ({timesteps}, {self.half_dim * 2})")
        time_embedding = time_embedding.view(timesteps, self.half_dim * 2)
        time_embedding = time_embedding.to(device=device)

        self.time_embedding = nn.Embedding.from_pretrained(
            time_embedding, freeze=True
        )

        self.embedding = nn.Sequential(
            nn.Linear(self.half_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.device = device
        self.embedding.to(device=device)


        nn.init.xavier_uniform_(self.embedding[0].weight)
        nn.init.xavier_uniform_(self.embedding[2].weight)
        self.embedding[0].bias.data.fill_(0)
        self.embedding[2].bias.data.fill_(0)

    def forward(self, t):
        t = t.to(device=self.device)
        return self.embedding(
            self.time_embedding(t)
        )