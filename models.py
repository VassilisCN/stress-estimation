import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameEncoderCNN(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(FrameEncoderCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, embed_dim, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.permute(0, 2, 1)  # [B, F, T] for Conv1d
        x = self.conv(x)        # [B, E, T]
        x = x.permute(0, 2, 1)  # [B, T, E]
        return x

class TransformerTemporalModel(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerTemporalModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

class AttentionMILHead(nn.Module):
    def __init__(self, embed_dim, num_classes=1):
        super(AttentionMILHead, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)  # num_classes should be defined externally

    def forward(self, x):
        # x: [B, T, E]
        attn_weights = self.attn(x)               # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * x, dim=1)  # [B, E]
        logits = self.classifier(context)         # [B, 1]
        return logits, attn_weights

class StressDetectionModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_classes=1):
        super(StressDetectionModel, self).__init__()
        self.encoder = FrameEncoderCNN(input_dim=input_dim, embed_dim=embed_dim)
        self.temporal_model = TransformerTemporalModel(embed_dim=embed_dim)
        self.head = AttentionMILHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x):
        # x: [B, T, F]
        x = self.encoder(x)              # [B, T, E]
        x = self.temporal_model(x)      # [B, T, E]
        logits, attn_weights = self.head(x)  # logits: [B, 1], attn: [B, T, 1]
        return logits.squeeze(-1), attn_weights.squeeze(-1)