import torch
import os
import torch.nn as nn
import numpy as np


# from ToolsMamba.my_vmamba import SS2D


class SpectralTokenEmbed(nn.Module):

    def __init__(self, in_chans, embed_dim, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        self.spectral_proj = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, 200, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.spectral_proj(x)  # [B, embed_dim, H, W]

        x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+H*W, embed_dim]

        N = x.shape[1]
        pos_embed = nn.functional.interpolate(
            self.pos_embed.transpose(1, 2), size=N, mode='linear', align_corners=False
        ).transpose(1, 2)
        x = x + pos_embed
        return x


def forward_dinov3_spectral(model, x, spectral_guidance=None):

    B = x.shape[0]

    x = model.spectral_embed(x)  # [B, N, D]
    N = x.shape[1]

    patch_N = N - 1
    H = W = int(patch_N ** 0.5)

    rope_sincos = model.rope_embed(H=H, W=W)

    for layer_idx, blk in enumerate(model.blocks):
        if layer_idx in [8, 12, 16] and spectral_guidance is not None:
            x = blk(x, rope_sincos, spectral_guidance)
        else:
            x = blk(x, rope_sincos)

    x = model.norm(x)
    return x


class BottleneckMLPAdapter(nn.Module):

    def __init__(self, dim, bottleneck_dim=64, spectral_channels=144):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.norm = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, dim)

        self.spec_proj = nn.Linear(spectral_channels, bottleneck_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)
        nn.init.trunc_normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)

        nn.init.trunc_normal_(self.up_proj.weight, std=1e-5)
        nn.init.zeros_(self.up_proj.bias)

        nn.init.zeros_(self.spec_proj.weight)
        nn.init.zeros_(self.spec_proj.bias)

    def forward(self, x, spec_vec=None):

        x = self.norm(x)
        x = self.down_proj(x)  # [B, N, bottleneck_dim]

        if spec_vec is not None:
            spec_bottleneck = self.spec_proj(spec_vec)  # [B, bottleneck_dim]
            x = x + spec_bottleneck.unsqueeze(1)

        x = self.act(x)
        x = self.up_proj(x)
        return x


def load_dinov3(repo_dir, weight_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    model = torch.hub.load(
        repo_dir, 'dinov3_vitl16', source='local', weights=weight_path
    )
    model = model.to(device)
    model.eval()
    return model


def load_dinov3_with_adapter(repo_dir, weight_path, in_chans, device="cuda" if torch.cuda.is_available() else "cpu",
                             use_gradient_checkpointing=False):

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    model = torch.hub.load(
        repo_dir, 'dinov3_vitl16', source='local', weights=weight_path
    )
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    spectral_embed = SpectralTokenEmbed(
        in_chans=in_chans,
        embed_dim=model.embed_dim,
        hidden_dim=256
    ).to(device)
    model.spectral_embed = spectral_embed
    print(f"已添加高光谱专属嵌入层，输入波段数: {in_chans}，输出维度: {model.embed_dim}")

    KEY_LAYERS = [4, 8, 12, 16]
    print(f"在关键层插入Adapter：{KEY_LAYERS}（共{len(KEY_LAYERS)}层）")
    adapted_blocks = []
    for layer_idx, block in enumerate(model.blocks):
        if layer_idx in KEY_LAYERS:
            dim = block.attn.qkv.in_features
            attn_adapter = BottleneckMLPAdapter(dim=dim, bottleneck_dim=128, spectral_channels=in_chans).to(device)
            ffn_adapter = BottleneckMLPAdapter(dim=dim, bottleneck_dim=128, spectral_channels=in_chans).to(device)

            class AdapterBlock(nn.Module):
                def __init__(self, attn_adapter, ffn_adapter, block):
                    super().__init__()
                    self.attn_adapter = attn_adapter
                    self.ffn_adapter = ffn_adapter
                    self.block = block

                def forward(self, x, rope_sincos, spectral_guidance=None):

                    x_norm1 = self.block.norm1(x)
                    attn_out = self.block.attn(x_norm1, rope=rope_sincos)


                    spec_vec = spectral_guidance.mean(dim=1) if spectral_guidance is not None else None
                    attn_adapt = self.attn_adapter(x_norm1, None)

                    x = x + self.block.ls1(attn_out + attn_adapt)

                    x_norm2 = self.block.norm2(x)
                    ffn_out = self.block.mlp(x_norm2)
                    ffn_adapt = self.ffn_adapter(x_norm2, spec_vec)

                    x = x + self.block.ls2(ffn_out + ffn_adapt)
                    return x

            adapted_blocks.append(AdapterBlock(attn_adapter, ffn_adapter, block))
        else:
            adapted_blocks.append(block)
    model.blocks = nn.ModuleList(adapted_blocks)
    return model


def extract_dinov3_features(model, input_image, device, feature_layer="norm"):
    input_image = input_image.to(device)
    model.eval()
    with torch.no_grad():
        x = forward_dinov3_spectral(model, input_image)
        B, N, C = x.shape
        patch_N = N - 1
        H = W = int(patch_N ** 0.5)

        spatial_features = x[:, 1:, :].reshape(B, H, W, C)
        features = spatial_features.cpu().numpy()
        spatial_shape = (H, W)
    return features, spatial_shape


def extract_dinov3_features_train(model, input_image, spectral_guidance=None):

    x = forward_dinov3_spectral(model, input_image, spectral_guidance)

    patch_tokens = x[:, 1:, :]
    features = torch.mean(patch_tokens, dim=1)
    return features
