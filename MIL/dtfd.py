"""Two-tier DTFD MaxMinS MIL head."""
import torch
from torch import nn
from torch.nn import functional as F


class DimReduction(nn.Module):
    def __init__(self, in_dim, reduced_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, reduced_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.fc1(x))


class AttentionGated(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(attention_dim, 1)

    def forward(self, x):
        scores = self.attention_weights(self.attention_V(x) * self.attention_U(x))
        return F.softmax(scores, dim=1)


class AttentionWithClassifier(nn.Module):
    def __init__(self, input_dim, attention_dim, num_classes):
        super().__init__()
        self.attention = AttentionGated(input_dim, attention_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(torch.sum(self.attention(x) * x, dim=1))


class DTFDMIL(nn.Module):
    """Distil each pseudo-bag's highest/lowest class-activation instances, then classify the combined bag."""
    def __init__(self, in_dim, hidden_dim=128, num_classes=2, m_dim=512, num_groups=4, split_mode='tensor_split'):
        super().__init__()
        if num_groups < 1 or split_mode not in ('chunk', 'tensor_split'):
            raise ValueError('Invalid DTFD pseudo-bag split configuration.')
        self.dim_reduction = DimReduction(in_dim, m_dim)
        self.attention = AttentionGated(m_dim, hidden_dim)
        self.classifier = nn.Linear(m_dim, num_classes)
        self.att_cls = AttentionWithClassifier(m_dim, hidden_dim, num_classes)
        self.num_groups, self.split_mode = num_groups, split_mode

    def forward(self, x):
        """First-tier predictions supply auxiliary losses; MaxMinS features feed the second tier.

        Ranking uses the final class's activation. PANDA RN18 uses torch.chunk, while the
        CAMELYON/foundation-model configuration uses torch.tensor_split.
        """
        split = torch.chunk if self.split_mode == 'chunk' else torch.tensor_split
        bags = split(x, min(self.num_groups, x.shape[1]), dim=1)
        distilled, predictions = [], []
        for bag in bags:
            reduced = self.dim_reduction(bag)
            weighted = reduced * self.attention(reduced)
            predictions.append(self.classifier(weighted.sum(dim=1)))
            cam = torch.einsum('bgf,cf->bgc', weighted, self.classifier.weight)
            order = torch.argsort(cam[:, :, -1], dim=1, descending=True)
            selected = torch.cat((order[:, :1], order[:, -1:]), dim=1)
            batch = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
            distilled.append(reduced[batch, selected])
        return self.att_cls(torch.cat(distilled, dim=1)), predictions
