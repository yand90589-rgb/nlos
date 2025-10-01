import torch
from torch import nn


class CAM(nn.Module):
    def __init__(self, inc, fusion='concat'):
        super().__init__()
        assert fusion in ['weight', 'adaptive', 'concat']
        self.fusion = fusion
        self.fusion_1 = nn.Conv3d(inc, inc, 1)
        self.fusion_2 = nn.Conv3d(inc, inc, 1)
        if self.fusion == 'adaptive':
            self.fusion_3 = nn.Conv3d(inc * 2, 2, 1)

        if self.fusion == 'concat':
            self.fusion_4 = nn.Conv3d(inc * 2, inc, 1)

    def forward(self, x1, x2):
        if self.fusion == 'weight':
            return self.fusion_1(x1) + self.fusion_2(x2)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(
                self.fusion_3(torch.cat([self.fusion_1(x1), self.fusion_2(x2)], dim=1)), dim=1)
            x1_weight, x2_weight = torch.split(fusion, [1, 1], dim=1)
            return x1 * x1_weight + x2 * x2_weight
        else:
            return self.fusion_4(torch.cat([self.fusion_1(x1), self.fusion_2(x2)], dim=1))


class Refinement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att_conv1 = nn.Conv3d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = nn.Conv3d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)

        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.refine_chann = dim
        self.refine = nn.Sequential(
            nn.Conv3d(self.refine_chann, self.refine_chann * 2, 7, stride=(1, 1, 1), padding=3, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv3d(self.refine_chann * 2, self.refine_chann * 2, 5, stride=(1, 1, 1), padding=2, dilation=1,
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv3d(self.refine_chann * 2, self.refine_chann, 3, stride=(1, 1, 1), padding=1, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=False))

        self.cam = CAM(inc=4, fusion='adaptive')

    def forward(self, x):
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)
        att = torch.cat([att1, att2], dim=1)
        avg_att = torch.mean(att, dim=1, keepdim=True)
        max_att, _ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:, 0, :, :, :].unsqueeze(1) + att2 * att[:, 1, :, :, :].unsqueeze(1)
        output1 = output + x
        output2 = self.refine(x)
        output = self.cam(output1, output2)
        return output


class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv3d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)

        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output


