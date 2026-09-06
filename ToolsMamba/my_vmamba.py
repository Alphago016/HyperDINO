import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath

from ToolsMamba.SSM import selective_scan


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # 1024×1=1024
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(
            torch.randn(4, self.dt_rank + 2 * self.d_state, self.d_inner, **factory_kwargs))
        self.dt_projs_weight = nn.Parameter(torch.randn(4, self.d_inner, self.dt_rank, **factory_kwargs))
        self.dt_projs_bias = nn.Parameter(torch.randn(4, self.d_inner, **factory_kwargs))

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.local_depth_conv = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=3, padding=1, groups=self.d_inner, bias=conv_bias
        )
        self.local_ln = nn.LayerNorm(self.d_inner)
        self.local_act = nn.SiLU()
        self.local_point_conv = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=1, bias=conv_bias
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):

        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            nn.init.constant_(dt_proj.weight, dt_init_std)

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):

        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner)
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):

        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):

        B, C, H, W = x.shape  # C = d_inner=1024
        L = H * W
        K = 4  # 4个方向：H→W、W→H、逆H→W、逆W→H

        x_hw = x.reshape(B, -1, L)
        x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous().reshape(B, -1, L)  # W→H方向：[B, 1024, L]
        x_hw_inv = torch.flip(x_hw, dims=[-1])  # 逆H→W方向
        x_wh_inv = torch.flip(x_wh, dims=[-1])  # 逆W→H方向
        xs = torch.stack([x_hw, x_wh, x_hw_inv, x_wh_inv], dim=1)  # (B, K, D_inner=1024, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)  # (B, K, dt_rank+2*d_state, L)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        # 处理delta（时间步长）
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)  # (B, K, D_inner=1024, L)
        dts = dts + self.dt_projs_bias.view(1, K, -1, 1)
        dts = F.softplus(dts)

        xs = xs.float().reshape(B * K, -1, L)  # (B*K, D_inner=1024, L)
        dts = dts.float().reshape(B * K, -1, L)  # (B*K, D_inner=1024, L)
        Bs = Bs.float().reshape(B * K, self.d_state, L)  # (B*K, d_state=16, L)
        Cs = Cs.float().reshape(B * K, self.d_state, L)  # (B*K, d_state=16, L)

        As = -torch.exp(self.A_logs.float()).reshape(self.d_inner, self.d_state)  # (D_inner=1024, d_state=16)
        Ds = self.Ds.float().reshape(self.d_inner)  # (D_inner=1024,)

        out_y = selective_scan(
            x=xs.permute(0, 2, 1),  # 适配selective_scan输入：(B*K, L, D_inner=1024)
            delta=dts.permute(0, 2, 1),  # (B*K, L, D_inner=1024)
            A=As,  # (D_inner=1024, d_state=16)
            B=Bs.permute(0, 2, 1),  # (B*K, L, d_state=16)
            C=Cs.permute(0, 2, 1),  # (B*K, L, d_state=16)
            D=Ds  # (D_inner=1024,)
        )
        out_y = out_y.permute(0, 2, 1).reshape(B, K, -1, L)  # 还原形状：(B, K, D_inner=1024, L)

        # 合并4个方向的结果
        y_hw = out_y[:, 0].reshape(B, self.d_inner, H, W)
        y_wh = out_y[:, 1].reshape(B, self.d_inner, W, H).transpose(2, 3)  # 还原W→H方向为H×W
        y_hw_inv = torch.flip(out_y[:, 2].reshape(B, self.d_inner, H, W), dims=[-1])  # 还原逆方向
        y_wh_inv = torch.flip(out_y[:, 3].reshape(B, self.d_inner, W, H), dims=[-1]).transpose(2, 3)

        return y_hw, y_wh, y_hw_inv, y_wh_inv

    def forward(self, x: torch.Tensor):

        B, H, W, C = x.shape  # C = d_model=1024

        # 输入投影 + 拆分x和z
        xz = self.in_proj(x)  # (B, H, W, 2*d_inner=2048)
        x, z = xz.chunk(2, dim=-1)  # (B, H, W, d_inner=1024) 各占一半

        # 深度卷积捕捉局部特征
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, d_inner=1024, H, W)
        x = self.act(self.conv2d(x))  # (B, 1024, H, W)

        # 4个方向的SSM处理
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4  # 融合4个方向结果：(B, 1024, H, W)

        y_enhanced = self.local_depth_conv(y)  # (B, 1024, H, W)

        y_enhanced = y_enhanced.permute(0, 2, 3, 1).contiguous()

        y_enhanced = self.local_ln(y_enhanced)

        y_enhanced = self.local_act(y_enhanced)

        y_enhanced = y_enhanced.permute(0, 3, 1, 2).contiguous()

        y_enhanced = self.local_point_conv(y_enhanced)

        y = y_enhanced + y

        y = y.permute(0, 2, 3, 1).contiguous()  # (B, H, W, d_inner=1024)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)  # (B, H, W, d_model=1024)

        if self.dropout is not None:
            out = self.dropout(out)

        return out
