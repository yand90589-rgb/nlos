import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import init
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce
from operator import mul


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class PatchEmbed3D(nn.Module):

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        x = self.norm(x)
        x = self.reduction(x)

        return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class WindowAttention3D(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class Pool(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(Pool, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool3d(kernel_size=(3, 17, 17), stride=(1, 8, 8), padding=(1, 8, 8)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool3d(kernel_size=(3, 9, 9), stride=(1, 4, 4), padding=(1, 4, 4)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool3d(kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AvgPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale_process = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(branch_planes * 4, branch_planes * 4, kernel_size=3, padding=1, groups=4, bias=False),
        )

        self.compression = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x, y):
        width = x.shape[-1]
        height = x.shape[-2]
        time = x.shape[-3]

        scale_list = []
        algc = True

        x_ = self.scale0(y)
        scale_list.append(F.interpolate(self.scale1(x), size=[time, height, width],
                                        mode='trilinear', align_corners=algc) + x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[time, height, width],
                                        mode='trilinear', align_corners=algc) + x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[time, height, width],
                                        mode='trilinear', align_corners=algc) + x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[time, height, width],
                                        mode='trilinear', align_corners=algc) + x_)

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return out


class ResConv3D(nn.Module):
    def __init__(self, nf0, inplace=False):
        super(ResConv3D, self).__init__()
        self.inplace = inplace
        self.tmp = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[1, 1, 1],
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace),
            nn.ReplicationPad3d(1),
            nn.Conv3d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[1, 1, 1],
                      bias=True))

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.temp_channels = self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.out_channels, 3, stride=(2, 2, 2), padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.conv1[0].bias, 0.0)

        self.conv2 = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 4, 4), bias=True,
                      dilation=(1, 4, 4)),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            ResConv3D(self.out_channels, False))

        self.conv3 = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 2, 2), bias=True,
                      dilation=(1, 2, 2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            ResConv3D(self.out_channels, False))

        self.conv4 = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True,
                      dilation=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            ResConv3D(self.out_channels, False))

    def forward(self, x):
        x = self.conv1(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.refine = nn.Sequential(
            nn.Conv3d(self.in_channels * 2, self.in_channels * 1, 3, stride=(1, 1, 1), padding=1, dilation=1,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            ResConv3D(self.in_channels * 1, False))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(self.in_channels, self.out_channels, (6, 6, 6), stride=(2, 2, 2), padding=(2, 2, 2),
                               bias=False),
            nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu')

        self.conv2 = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 4, 4), bias=True,
                      dilation=(1, 4, 4)),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            ResConv3D(self.out_channels, False))

        self.conv3 = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 2, 2), bias=True,
                      dilation=(1, 2, 2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            ResConv3D(self.out_channels, False))

        self.conv4 = nn.Sequential(
            nn.Conv3d(self.out_channels, self.out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True,
                      dilation=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            ResConv3D(self.out_channels, False))

    def forward(self, x, skip_x):
        xc = torch.cat([x, skip_x], dim=1)
        x = self.refine(xc)
        x = self.conv1(x)
        return x

class Attention3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder1 = DownBlock(in_channels=self.in_channels, out_channels=self.in_channels * 2)
        self.encoder2 = DownBlock(in_channels=self.in_channels * 2, out_channels=self.in_channels * 4)
        self.attn = WindowAttention3D(
            dim=4, window_size=(4, 4, 4), num_heads=4,
            qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0)

        self.decoder2 = UpBlock(in_channels=self.in_channels * 4, out_channels=self.in_channels * 2)
        self.decoder1 = UpBlock(in_channels=self.in_channels * 2, out_channels=self.out_channels)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)

        B, C, D, H, W = enc2.shape
        window_size = (4, 4, 4)
        enc3 = rearrange(enc2, 'b c d h w -> b d h w c')
        x_windows = window_partition(enc3, window_size)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        x = window_reverse(attn_windows, window_size, B, D, H, W)
        base = rearrange(x, 'b d h w c -> b c d h w')

        dec2 = self.decoder2(base, enc2)
        dec1 = self.decoder1(dec2, enc1)
        return dec1