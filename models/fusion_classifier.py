import torch
import torch.nn as nn
import numpy as np
from ToolsMamba.SSM import SSM


class SpectralBranch(nn.Module):

    def __init__(self, num_channels=3, output_dim=1024, hidden_channels=24,
                 kernel_size=5, num_res_blocks=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.channel_mix = nn.Conv1d(num_channels, hidden_channels, kernel_size=1)

        self.ssm = SSM(
            in_features=hidden_channels,
            dt_rank=2,
            dim_inner=hidden_channels,
            d_state=8
        )
        self.act = nn.GELU()
        self.query_proj = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, output_dim)
        self.out_act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.channel_mix.weight, std=0.02)
        nn.init.zeros_(self.channel_mix.bias)
        nn.init.trunc_normal_(self.query_proj.weight, std=0.02)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.trunc_normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.channel_mix(x)  # [B, hidden_channels, C]

        x = x.transpose(1, 2)

        residual = x
        x = self.ssm(x)
        x = self.act(x)
        x = x + residual

        x = x.transpose(1, 2)  # [B, hidden_channels, C]


        g = torch.mean(x, dim=-1)
        q = self.query_proj(g)
        attn = torch.einsum('bc,bcl->bl', q, x)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bl,bcl->bc', attn, x)
        out = self.out_proj(out)
        out = self.out_act(out)
        return out


class ScaleAttention(nn.Module):
    def __init__(self, feat_dim=1024, spatial_size=14):
        super().__init__()
        self.feat_dim = feat_dim
        self.spatial_size = spatial_size
        self.scale_weight = nn.Parameter(torch.tensor([0.6, 0.4]))

    def forward(self, feat_s1, feat_s2, spatial_shapes):
        B = feat_s1.shape[0]
        alpha, beta = torch.softmax(self.scale_weight, dim=0)
        fused_feat = alpha * feat_s1 + beta * feat_s2
        return fused_feat, (self.spatial_size, self.spatial_size)


class SpectralGuidedChannelAttention(nn.Module):

    def __init__(self, feat_dim=1024, reduction=8):
        super().__init__()
        self.feat_dim = feat_dim
        hidden_dim = feat_dim // reduction

        self.spec_norm = nn.LayerNorm(feat_dim)
        self.gamma_fc1 = nn.Linear(feat_dim, hidden_dim)
        self.gamma_fc2 = nn.Linear(hidden_dim, feat_dim)
        self.gamma_act = nn.GELU()
        self.gate_proj = nn.Linear(feat_dim, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.gamma_fc1.weight, std=0.02)
        nn.init.zeros_(self.gamma_fc1.bias)
        nn.init.trunc_normal_(self.gamma_fc2.weight, std=0.02)
        nn.init.constant_(self.gamma_fc2.bias, 2.0)
        nn.init.trunc_normal_(self.gate_proj.weight, std=0.02)
        nn.init.constant_(self.gate_proj.bias, -5.0)

    def forward(self, spatial_feat, spectral_feat):
        spec_norm = self.spec_norm(spectral_feat)
        gamma = self.gamma_fc1(spec_norm)
        gamma = self.gamma_act(gamma)
        gamma = self.gamma_fc2(gamma)
        gamma = torch.sigmoid(gamma)
        gate = torch.sigmoid(self.gate_proj(spec_norm))
        spatial_out = spatial_feat + gate * spatial_feat * (gamma - 1.0)
        return spatial_out


class SpatialGuidedSpectralAttention(nn.Module):

    def __init__(self, feat_dim=1024, reduction=8):
        super().__init__()
        self.feat_dim = feat_dim
        hidden_dim = feat_dim // reduction
        self.spatial_norm = nn.LayerNorm(feat_dim)
        self.gamma_fc1 = nn.Linear(feat_dim, hidden_dim)
        self.gamma_fc2 = nn.Linear(hidden_dim, feat_dim)
        self.gamma_act = nn.GELU()
        self.gate_proj = nn.Linear(feat_dim, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.gamma_fc1.weight, std=0.02)
        nn.init.zeros_(self.gamma_fc1.bias)
        nn.init.trunc_normal_(self.gamma_fc2.weight, std=0.02)
        nn.init.constant_(self.gamma_fc2.bias, 2.0)
        nn.init.trunc_normal_(self.gate_proj.weight, std=0.02)
        nn.init.constant_(self.gate_proj.bias, -5.0)

    def forward(self, spectral_feat, spatial_feat):
        spatial_norm = self.spatial_norm(spatial_feat)
        gamma = self.gamma_fc1(spatial_norm)
        gamma = self.gamma_act(gamma)
        gamma = self.gamma_fc2(gamma)
        gamma = torch.sigmoid(gamma)
        gate = torch.sigmoid(self.gate_proj(spatial_norm))
        spectral_out = spectral_feat + gate * spectral_feat * (gamma - 1.0)
        return spectral_out


class ImprovedClassifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class HSIClassifier(nn.Module):
    def __init__(self, spectral_dim=None, num_classes=16):
        super().__init__()

        self.spectral_branch = SpectralBranch(
            num_channels=3,
            output_dim=1024,
            hidden_channels=24,
            kernel_size=5,
            num_res_blocks=1
        )

        self.sgca = SpectralGuidedChannelAttention(feat_dim=1024, reduction=8)

        self.sgsa = SpatialGuidedSpectralAttention(feat_dim=1024, reduction=8)

        self.fusion_weight = nn.Parameter(torch.tensor([1.0, 1.0]))

        self.classifier = ImprovedClassifier(in_dim=2048, num_classes=num_classes)

    def forward(self, feat_spatial, spectral_feat):

        spectral_feat = self.spectral_branch(spectral_feat)

        spatial_modulated = self.sgca(feat_spatial, spectral_feat)
        spectral_modulated = self.sgsa(spectral_feat, feat_spatial)

        fused_feat = torch.cat([
            self.fusion_weight[0] * spatial_modulated,
            self.fusion_weight[1] * spectral_modulated
        ], dim=-1)
        outputs = self.classifier(fused_feat)
        return outputs
