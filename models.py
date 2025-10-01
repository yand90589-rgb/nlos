import numpy as np
import lct
import fk
import torch
from torch import nn
from STRM import Pool
from STRM import Attention3D
from DERM import Fusion, Refinement
import torch.nn.functional as F


class Interpsacle2d(nn.Module):

    def __init__(self, factor=2, gain=1, align_corners=False):
        super(Interpsacle2d, self).__init__()
        self.gain = gain
        self.factor = factor
        self.align_corners = align_corners

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain

        x = nn.functional.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=self.align_corners)

        return x


class ResConv2D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv2D, self).__init__()

        self.tmp = nn.Sequential(

            nn.ReplicationPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[3, 3],
                      padding=0,
                      stride=[1, 1],
                      bias=True),

            nn.LeakyReLU(negative_slope=0.2, inplace=inplace),

            nn.ReplicationPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[3, 3],
                      padding=0,
                      stride=[1, 1],
                      bias=True),
        )

        self.inplace = inplace

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re


class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()

    def forward(self, x):
        x, idx = x.max(2)
        d = x.size(2)
        depth = (d - 1 - idx.float()) / (d - 1)
        out = torch.cat([x, depth], dim=1)
        return out

class ResConv3D(nn.Module):
    def __init__(self, nf0, inplace=False):
        super(ResConv3D, self).__init__()
        self.tmp = nn.Sequential(

            nn.ReplicationPad3d(1),
            nn.Conv3d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[1, 1, 1],
                      bias=True),

            nn.LeakyReLU(negative_slope=0.2, inplace=inplace),

            nn.ReplicationPad3d(1),
            nn.Conv3d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[1, 1, 1],
                      bias=True),
        )

        self.inplace = inplace

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re


class Transient2volumn_gray(nn.Module):
    def __init__(self, nf0, in_channels, \
                 norm=nn.InstanceNorm3d):
        super(Transient2volumn_gray, self).__init__()

        weights = np.zeros((in_channels, in_channels, 3, 3, 3), dtype=np.float32)
        weights[:, :, 1:, 1:, 1:] = 1.0
        tfweights = torch.from_numpy(weights / np.sum(weights))
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)

        self.conv1 = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[2, 2, 2],
                      bias=True),
            ResConv3D(nf0 * 1, inplace=False),
            ResConv3D(nf0 * 1, inplace=False)
        )

    def forward(self, x0):
        x0_conv = F.conv3d(x0, self.weights, \
                           bias=None, stride=2, padding=1, dilation=1, groups=1)
        x1 = self.conv1(x0)
        re = torch.cat([x0_conv, x1], dim=1)
        return re

class Rendering(nn.Module):

    def __init__(self, nf0, out_channels, \
                 norm=nn.InstanceNorm2d, isdep=False):
        super(Rendering, self).__init__()

        self.out_channels = out_channels
        weights = np.zeros((out_channels, out_channels * 2, 1, 1), dtype=np.float32)
        if isdep:
            weights[:, out_channels:, :, :] = 1.0
        else:
            weights[:, :out_channels, :, :] = 1.0
        tfweights = torch.from_numpy(weights)
        tfweights.requires_grad = True
        self.weights = nn.Parameter(tfweights)

        self.resize = Interpsacle2d(factor=2, gain=1, align_corners=False)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 1, inplace=False),
            ResConv2D(nf0 * 1, inplace=False),
        )

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 1 + out_channels,
                      nf0 * 2,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
            ResConv2D(nf0 * 2, inplace=False),
            ResConv2D(nf0 * 2, inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(nf0 * 2,
                      out_channels,
                      kernel_size=3,
                      padding=0,
                      stride=1,
                      bias=True),
        )

    def forward(self, x0):
        dim = x0.shape[1] // 2
        x0_im = x0[:, 0:self.out_channels, :, :]
        x0_dep = x0[:, dim:dim + self.out_channels, :, :]
        x0_raw_128 = torch.cat([x0_im, x0_dep], dim=1)
        x0_raw_256 = self.resize(x0_raw_128)
        x0_conv_256 = F.conv2d(x0_raw_256, self.weights, \
                               bias=None, stride=1, padding=0, dilation=1, groups=1)
        x1 = self.conv1(x0)
        x1_up = self.resize(x1)

        x2 = torch.cat([x0_conv_256, x1_up], dim=1)
        x2 = self.conv2(x2)

        re = x0_conv_256 + 1 * x2

        return re


class Model(nn.Module):
    def __init__(self, basedim=3, in_ch=1, out_ch=1, spatial=64, tlen=256, bin_len=0.02, views=1, wall_size=2,
                 sp_ds_scale=1):
        super(Model, self).__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.spatial = spatial
        self.tlen = tlen
        self.bin_len = bin_len
        self.views = views
        self.wall_size = wall_size
        self.up_num = int(np.log2(sp_ds_scale))
        self.cs_ch = 4
        self.sp_ds_scale = sp_ds_scale
        self.ppm = Pool(inplanes=1, branch_planes=4, outplanes=1)
        self.unet = Attention3D(in_channels=1, out_channels=1)

        self.down_net = Transient2volumn_gray(nf0=basedim, in_channels=self.in_channels)

        self.tra2vlo = lct.lct(spatial=self.spatial, crop=self.tlen, wall_size=self.wall_size,
                                    bin_len=self.bin_len, method='lct', dnum=basedim + self.in_channels)
        self.tra3vlo = fk.lct_fk(spatial=self.spatial, crop=self.tlen, bin_len=self.bin_len,
                                      dnum=basedim + self.in_channels)
        refine_chann = basedim + self.in_channels

        self.refine_lct = Refinement(dim=4)
        self.refine_fk = Refinement(dim=4)

        self.diff = Fusion(dim=4)

        self.visnet = Projector()
        self.rendernet = Rendering(nf0=(basedim * 1 + self.in_channels) * 2, out_channels=self.out_channels)
        self.depnet = Rendering(nf0=(basedim * 1 + self.in_channels) * 2, out_channels=self.out_channels, isdep=True)

    def normalize(self, data_bxcxdxhxw):
        b, c, d, h, w = data_bxcxdxhxw.shape
        data_bxcxk = data_bxcxdxhxw.reshape(b, c, -1)
        data_min = data_bxcxk.min(2, keepdim=True)[0]
        data_zmean = data_bxcxk - data_min
        data_max = data_zmean.max(2, keepdim=True)[0]
        data_norm = data_zmean / (data_max + 1e-15)

        return data_norm.view(b, c, d, h, w)

    def forward(self, inputs, target=None, targetd=None):

        if self.sp_ds_scale >= 1:
            b, c, t, h, w = inputs.shape
            ds_mea_reshape = inputs.reshape(-1, t, h, w)
            ds_mea_reshape = ds_mea_reshape[:, :, self.sp_ds_scale // 2::self.sp_ds_scale,
                             self.sp_ds_scale // 2::self.sp_ds_scale]
            simple_up = F.interpolate(ds_mea_reshape, size=(128,) * 2, mode='bilinear',
                                      align_corners=True)
            f = simple_up.reshape(b, c, t, 128, 128)

        sig_exp = self.ppm(f, inputs)
        sig_exp = self.unet(sig_exp)

        inputs = self.normalize(sig_exp)
        fea = self.down_net(inputs)

        vlo_lct = self.tra2vlo(fea, [0, 0, 0], [self.tlen, self.tlen, self.tlen])

        zdim = vlo_lct.shape[2]
        zdimnew = zdim * 100 // 128
        vlo_lct = vlo_lct[:, :, :zdimnew]
        vlo_lct = nn.ReLU()(vlo_lct)
        vlo_lct = self.normalize(vlo_lct)
        results_lct = vlo_lct
        results_lct = self.refine_lct(results_lct)

        vlo_fk = self.tra3vlo(fea, [0, 0, 0], [self.tlen, self.tlen, self.tlen])

        zdim = vlo_fk.shape[2]
        zdimnew = zdim * 100 // 128
        vlo_fk = vlo_fk[:, :, :zdimnew]
        vlo_fk = nn.ReLU()(vlo_fk)
        vlo_fk = self.normalize(vlo_fk)
        results_fk = vlo_fk
        results_fk = self.refine_fk(results_fk)

        results = self.diff(results_lct, results_fk)
        raw = self.visnet(results)

        rendered_img = self.rendernet(raw)
        rendered_img = rendered_img.reshape(inputs.shape[0], -1, *rendered_img.shape[-3:])
        rendered_depth = self.depnet(raw)
        rendered_depth = rendered_depth.reshape(inputs.shape[0], -1, *rendered_img.shape[-3:])
        rendered_depth = torch.mean(rendered_depth, dim=2, keepdim=True)

        if target is not None:
            target = target.cuda(non_blocking=True)
            targetd = targetd.cuda(non_blocking=True)
            return sig_exp, results, rendered_img, target, rendered_depth, targetd
        else:
            return sig_exp, torch.squeeze(rendered_img, 1)
