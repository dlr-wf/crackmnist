import torch
from torch import nn
from torch.nn import functional as F


class UNet(nn.Module):
    """UNet with 4 blocks like originally proposed by Ronneberger et al."""

    def __init__(self, in_ch=2, out_ch=1, init_features=64, dropout_prob=0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.init_features = init_features
        self.dropout_prob = dropout_prob

        self.inc = DoubleConv(self.in_ch, self.init_features)
        self.down1 = Down(self.init_features, self.init_features * 2)
        self.down2 = Down(self.init_features * 2, self.init_features * 2)

        self.base = Base(self.init_features * 2, dropout_prob=dropout_prob)

        self.up3 = Up(self.init_features * 4, self.init_features * 1)
        self.up4 = Up(self.init_features * 2, self.init_features)
        self.outc = OutConv(self.init_features, self.out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x3 = self.base(x3)

        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return torch.sigmoid(x)


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.doubleconv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.doubleconv(x)
        return x


class Base(nn.Module):
    def __init__(self, channels, dropout_prob):
        super().__init__()

        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_ch // 2, in_ch // 2, kernel_size=2, stride=2
            )

        self.doubleconv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
        )

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        # x = x1 + x2
        x = self.doubleconv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
