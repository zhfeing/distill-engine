from torch import nn
import torch.nn.functional as F
import torch


class ResBlock(nn.Module):
    def __init__(self, in_channels, with_subsample):
        super(ResBlock, self).__init__()
        self._in_channels = in_channels
        self._with_subsample = with_subsample

        if with_subsample:
            self._out_channels = 2*in_channels
        else:
            self._out_channels = in_channels

        self._conv_1 = None
        self._conv_2 = None
        self._short_cut = None
        self.build()

    def build(self):
        if self._with_subsample:
            self._conv_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._in_channels,
                    out_channels=self._out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(self._out_channels),
                nn.ReLU()
            )
        else:
            self._conv_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._in_channels,
                    out_channels=self._out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2d(self._out_channels),
                nn.ReLU()
            )

        self._conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self._out_channels,
                out_channels=self._out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self._out_channels)
        )

        if self._with_subsample:
            self._short_cut = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._in_channels,
                    out_channels=self._out_channels,
                    kernel_size=1,
                    stride=2
                ),
                nn.BatchNorm2d(self._out_channels)
            )
        else:
            self._short_cut = nn.Sequential()

    def forward(self, x):
        short_cut = self._short_cut(x)
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = F.relu(x + short_cut)
        return x


class Resnet(nn.Module):
    def __init__(self, n, in_channels, channel_base):
        super(Resnet, self).__init__()
        self._n = n
        self._in_channels = in_channels
        self._channel_base = channel_base
        self._conv_1 = None
        self._middle = None
        self._classifier = None
        self.build()

    def build(self):
        self._conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self._in_channels,
                out_channels=self._channel_base,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self._channel_base),
            nn.ReLU()
        )
        self._middle = nn.Sequential()
        in_channel = self._channel_base

        for i in range(3):
            if i == 0:
                self._middle.add_module(
                    'conv_{}_{}'.format(i, 0),
                    ResBlock(in_channel, False)
                )
            else:
                self._middle.add_module(
                    'conv_{}_{}'.format(i, 0),
                    ResBlock(in_channel, True)
                )
                in_channel *= 2

            for j in range(1, self._n):
                self._middle.add_module(
                    'conv_{}_{}'.format(i, j),
                    ResBlock(in_channel, False)
                )

        self._classifier = nn.Sequential(
            nn.Linear(in_channel, 10)
        )

    def forward(self, x):
        x = self._conv_1(x)
        x = self._middle(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self._classifier(x)
        return x


def my_resnet():
    model = Resnet(
        n=5,
        in_channels=3,
        channel_base=16
    )
    return model


def my_test():
    import numpy as np

    resnet_model = my_resnet()
    params = list(resnet_model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
        # print(l)
    k = format(k, ',')
    print("total parameters: " + k)

    x = np.random.random([1, 3, 32, 32])
    x = torch.Tensor(x)
    y = resnet_model(x)
    print(y.size())


if __name__ == "__main__":
    my_test()
