from torch.nn.utils import weight_norm, spectral_norm
import torch
from torch import nn
import torch.nn.functional as F

class MRF(nn.Module):
    def __init__(self, dim, kernel_size=3, deletions= (1, 3, 5), leaky_relu = 0.1):
        super(MRF, self).__init__()

        self.leaky_relu = leaky_relu
        self.conv1 = nn.ModuleList([
            weight_norm(nn.Conv1d(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = (kernel_size * deletions[0] - deletions[0]) // 2, dilation = deletions[0])),
            weight_norm(nn.Conv1d(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = (kernel_size * 1 - 1) // 2, dilation = 1))])

        self.conv2 = nn.ModuleList([
            weight_norm(nn.Conv1d(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = (kernel_size * deletions[1] - deletions[1]) // 2, dilation = deletions[1])),
            weight_norm(nn.Conv1d(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = (kernel_size * 1 - 1) // 2, dilation = 1))])

        self.conv3 = nn.ModuleList([
            weight_norm(nn.Conv1d(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = (kernel_size * deletions[2] - deletions[2]) // 2, dilation = deletions[2])),
            weight_norm(nn.Conv1d(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = (kernel_size * 1 - 1) // 2, dilation = 1))])


    def forward(self, x):
        for conv in (self.conv1, self.conv2, self.conv3):
            for index, layer in enumerate(conv):
                x_pr = F.leaky_relu(x if index == 0 else x_pr, self.leaky_relu)
                x_pr = layer(x_pr)
            x = x + x_pr
        return x


class Generator(nn.Module):
    def __init__(self,leaky_relu = 0.1):
        super(Generator, self).__init__()
        upsample_rates = [8,8,2,2]
        upsample_kernel_sizes = [16,16,4,4]
        upsample_initial_channel = 128
        resblock_kernel_sizes = [3,7,11]
        resblock_dilation_sizes =  [[1,3,5], [1,3,5], [1,3,5]]
        self.len_upsample = len(upsample_rates)
        self.len_blocks = len(resblock_kernel_sizes)

        self.conv = nn.Conv1d(in_channels = 80, out_channels = upsample_initial_channel, kernel_size = 7, stride = 1, padding = 3)


        ls = []
        for i in range(self.len_upsample):
            indim = upsample_initial_channel // (2 ** i),
            outdim = upsample_initial_channel // (2 ** (i + 1))
            kernal = upsample_kernel_sizes[i]
            stride = upsample_rates[i]
            ls += [
                nn.LeakyReLU(leaky_relu),
                weight_norm(nn.ConvTranspose1d(indim[0], outdim, kernal, stride, (kernal - stride) //2))]
        self.generator = nn.ModuleList(ls)


        ls_MRF_bloacks = []
        for i in range(self.len_upsample):
            dim = upsample_initial_channel // 2 ** (i + 1)
            for j in range(self.len_blocks):
                kernal = resblock_kernel_sizes[j]
                stride = resblock_dilation_sizes[j]
                ls_MRF_bloacks += [
                    MRF(dim, kernal, stride )
                ]
        self.resblocks = nn.ModuleList(ls_MRF_bloacks)
        self.conv_post = weight_norm(nn.Conv1d(dim, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv(x)
        for i in range(self.len_upsample):
            x = self.generator[i * 2](x)
            x = self.generator[i * 2 + 1](x)
            for j in range(self.len_blocks):
                if j == 0:
                    xs = self.resblocks[i*self.len_blocks+j](x)
                else:
                    xs += self.resblocks[i*self.len_blocks+j](x)
            x = xs / self.len_blocks
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class MSD_block(nn.Module):
    def __init__(self, norm, leaky_relu = 0.1):
        super(MSD_block, self).__init__()

        self.conv = nn.ModuleList([
            norm(nn.Conv1d(in_channels = 1, out_channels = 128, kernel_size = 15, stride = 1, padding = 7)),
            norm(nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 41, stride = 2, padding = 20, groups = 4)),
            norm(nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 41, stride = 2, padding = 20, groups = 16)),
            norm(nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 41, stride = 4, padding = 20, groups = 16)),
            norm(nn.Conv1d(in_channels = 512, out_channels = 1024, kernel_size = 41, stride = 4, padding = 20, groups = 16)),
            norm(nn.Conv1d(in_channels = 1024, out_channels = 1024, kernel_size = 41, stride = 1, padding = 20, groups = 16)),
            norm(nn.Conv1d(in_channels = 1024, out_channels = 1024, kernel_size = 5, stride = 1, padding = 2)) ])
        self.last_layer = norm(nn.Conv1d(in_channels = 1024, out_channels = 1, kernel_size = 3, stride = 1, padding = 1))
        self.leaky_relu = nn.LeakyReLU(leaky_relu)


    def forward(self, x):
        feature_maps = []
        for layer in self.conv:
            x = layer(x)
            x = self.leaky_relu(x)
            feature_maps.append(x)
        x = self.last_layer(x)
        feature_maps.append(x)

        return x.flatten(1, -1), feature_maps

class MSD(nn.Module):
    def __init__(self):
        super(MSD, self).__init__()

        self.descr = nn.ModuleList([
            MSD_block(spectral_norm),
            MSD_block(weight_norm),
            MSD_block(weight_norm)])

        self.avgpools = nn.ModuleList([
            nn.AvgPool1d(kernel_size = 4, stride = 2, padding = 2),
            nn.AvgPool1d(kernel_size = 4, stride = 2, padding = 2)
        ])


    def forward(self, y, y_head):
        y_outputs, y_head_outputs = [], []
        y_feature_maps, y_head_feature_maps = [], []
        for index, layer in enumerate(self.descr):
            if index != 0:
                y = self.avgpools[index - 1](y)
                y_head = self.avgpools[index - 1](y_head)

            y_out, y_feature_map = layer(y)
            y_head_out, y_head_feature_map = layer(y_head)
            y_outputs.append(y_out)
            y_head_outputs.append(y_head_out)
            y_feature_maps.append(y_feature_map)
            y_head_feature_maps.append(y_head_feature_map)

        return y_outputs, y_head_outputs, y_feature_maps, y_head_feature_maps


class MPD_block(nn.Module):
    def __init__(self, period, norm, leaky_relu = 0.1):
        super(MPD_block, self).__init__()
        self.period = period
        self.conv = nn.ModuleList([
            norm(nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (5, 1), stride = (3, 1), padding = ((5 * 1 - 1) // 2, 0))), #diferent weight 64?
            norm(nn.Conv2d(in_channels = 32, out_channels = 128, kernel_size = (5, 1), stride = (3, 1), padding = ((5 * 1 - 1) // 2, 0))),
            norm(nn.Conv2d(in_channels = 128, out_channels = 512, kernel_size = (5, 1), stride = (3, 1), padding = ((5 * 1 - 1) // 2, 0))),
            norm(nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = (5, 1), stride = (3, 1), padding = ((5 * 1 - 1) // 2, 0))),
            norm(nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = (5, 1), stride = 1, padding = (2, 0)))
        ])
        self.leaky_relu = nn.LeakyReLU(leaky_relu)
        self.last_layer = norm(nn.Conv2d(in_channels = 1024, out_channels = 1, kernel_size = (3, 1), stride = 1, padding = (1, 0)))
    def forward(self, x):
        feature_maps = []

        B, C, T = x.shape
        pading = self.period - (T % self.period)
        T = T + pading
        x = F.pad(x, (0, pading), "reflect")
        x = x.view(B, C, T // self.period, self.period)

        for layer in self.conv:
            x = layer(x)
            x = self.leaky_relu(x)
            feature_maps.append(x)
        x = self.last_layer(x)
        feature_maps.append(x)

        return x.flatten(1, -1), feature_maps


class MPD(nn.Module):
    def __init__(self):
        super(MPD, self).__init__()

        self.descr = nn.ModuleList([
            MPD_block(2, weight_norm),
            MPD_block(3, weight_norm),
            MPD_block(5, weight_norm),
            MPD_block(7, weight_norm),
            MPD_block(11, weight_norm),
        ])


    def forward(self, y, y_head):
        y_outputs, y_head_outputs = [], []
        y_feature_maps, y_head_feature_maps = [], []
        for index, layer in enumerate(self.descr):
            y_out, y_feature_map = layer(y)
            y_head_out, y_head_feature_map = layer(y_head)
            y_outputs.append(y_out)
            y_head_outputs.append(y_head_out)
            y_feature_maps.append(y_feature_map)
            y_head_feature_maps.append(y_head_feature_map)

        return y_outputs, y_head_outputs, y_feature_maps, y_head_feature_maps
