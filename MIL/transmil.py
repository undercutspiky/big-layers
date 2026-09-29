import numpy as np
import torch
from .nystrom_attention import NystromAttention
from torch import nn


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    """Two attention blocks with PPEG and a learned class token."""
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=hidden_dim)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.num_classes = num_classes
        self.layer1 = TransLayer(dim=hidden_dim)
        self.layer2 = TransLayer(dim=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x):
        # x.shape: [B, n, in_dim]
        x = self._fc1(x)  # [B, n, hidden_dim]

        # Repeat the first tokens to form the square spatial grid used by PPEG.
        H = x.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        x = torch.cat([x, x[:, :add_length, :]], dim=1)  # [B, N, hidden_dim]

        # ---->cls_token
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Self-attention, positional depthwise convolutions, then self-attention again.
        x = self.layer1(x)  # [B, N, hidden_dim]

        # ---->PPEG
        x = self.pos_layer(x, _H, _W)  # [B, N, hidden_dim]

        # ---->Translayer x2
        x = self.layer2(x)  # [B, N, hidden_dim]

        # ---->cls_token
        x = self.norm(x)[:, 0]

        # ---->predict
        logits = self._fc2(x)  # [B, n_classes]

        return logits, None
