from torch import nn
import torch.nn.functional as F
import torch


def get_block_output_size(channel_list):
    size = channel_list['#1x1'] + channel_list['#3x3'] + channel_list['#5x5'] + channel_list['pool proj']
    return size


def my_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    seq_model = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return seq_model


class InceptionBlock(nn.Module):
    def __init__(self, channel_list):
        super(InceptionBlock, self).__init__()
        self.channel_list = channel_list
        # path 1
        self._path_1 = nn.Sequential()
        self._path_1.add_module(
            'path_1_conv_1x1',
            my_conv(
                in_channels=channel_list['input'],
                out_channels=channel_list['#1x1'],
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

        # path 2
        self._path_2 = nn.Sequential()
        self._path_2.add_module(
            'path_2_conv_1x1',
            my_conv(
                in_channels=channel_list['input'],
                out_channels=channel_list['#3x3 reduce'],
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        self._path_2.add_module(
            'path_2_conv_3x3',
            my_conv(
                in_channels=channel_list['#3x3 reduce'],
                out_channels=channel_list['#3x3'],
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # path 3
        self._path_3 = nn.Sequential()
        self._path_3.add_module(
            'path_3_conv_1x1',
            my_conv(
                in_channels=channel_list['input'],
                out_channels=channel_list['#5x5 reduce'],
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        self._path_3.add_module(
            'path_3_conv_5x5',
            my_conv(
                in_channels=channel_list['#5x5 reduce'],
                out_channels=channel_list['#5x5'],
                kernel_size=5,
                stride=1,
                padding=2
            )
        )

        # path 4
        self._path_4 = nn.Sequential()
        self._path_4.add_module(
            'path_4_max_pool',
            nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        self._path_4.add_module(
            'path_4_conv_1x1',
            my_conv(
                in_channels=channel_list['input'],
                out_channels=channel_list['pool proj'],
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, x):
        x1 = self._path_1(x)
        x2 = self._path_2(x)
        x3 = self._path_3(x)
        x4 = self._path_4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class GoogLeNet(nn.Module):
    def __init__(self, input_channel, step_1_channel, class_num, channel_lists):
        super(GoogLeNet, self).__init__()
        self._input_channel = input_channel
        self._step_1_channel = step_1_channel
        self._class_num = class_num
        self._channel_list = channel_lists
        self._inception_block_num = len(channel_lists)
        self._inception_blocks = list()
        self._step_1 = None
        self._inception_3 = None
        self._inception_4a = None
        self._inception_4bcd = None
        self._inception_4e = None
        self._inception_5 = None
        self._classifier_main = None
        self._classifier_aux_1 = None
        self._classifier_aux_2 = None

        self.build_net()

    def build_net(self):
        for ls in self._channel_list:
            self._inception_blocks.append(InceptionBlock(ls))

        self._step_1 = nn.Sequential(
            my_conv(
                in_channels=self._input_channel,
                out_channels=self._step_1_channel,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            my_conv(
                in_channels=self._step_1_channel,
                out_channels=self._step_1_channel,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self._inception_3 = nn.Sequential(
            self._inception_blocks[0],
            self._inception_blocks[1],
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self._inception_4a = self._inception_blocks[2]

        self._inception_4bcd = nn.Sequential(
            self._inception_blocks[3],
            self._inception_blocks[4],
            self._inception_blocks[5]
        )

        self._inception_4e = nn.Sequential(
            self._inception_blocks[6],
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self._inception_5 = nn.Sequential(
            self._inception_blocks[7],
            self._inception_blocks[8]
        )

        self._classifier_main = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(get_block_output_size(self._channel_list[8]), self._class_num),
        )

        self._classifier_aux_1 = nn.Sequential(
            nn.Linear(get_block_output_size(self._channel_list[2]), 64),
            nn.Dropout(0.7),
            nn.Linear(64, self._class_num)
        )
        
        self._classifier_aux_2 = nn.Sequential(
            nn.Linear(get_block_output_size(self._channel_list[5]), 64),
            nn.Dropout(0.7),
            nn.Linear(64, self._class_num)
        )

    def forward(self, x):
        x = self._step_1(x)
        x = self._inception_3(x)

        x = self._inception_4a(x)

        # to aux classifier
        x1 = F.avg_pool2d(x, x.size()[2:])
        x1 = x1.view(x1.size(0), -1)
        x1 = self._classifier_aux_1(x1)

        # main path
        x = self._inception_4bcd(x)

        # to aux classifier
        x2 = F.avg_pool2d(x, x.size()[2:])
        x2 = x2.view(x2.size(0), -1)
        x2 = self._classifier_aux_2(x2)

        # main path
        x = self._inception_4e(x)
        x = self._inception_5(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self._classifier_main(x)
        return x, x1, x2


def make_channel_list(*args):
    channel_list = {
            'input': args[0],
            '#1x1': args[1],
            '#3x3 reduce': args[2],
            '#3x3': args[3],
            '#5x5 reduce': args[4],
            '#5x5': args[5],
            'pool proj': args[6]
        }
    return channel_list


def my_googLeNet():
    import numpy as np
    step_1_channel = 32

    parameters = [
        [32, 48, 64, 8, 16, 16],
        [64, 64, 96, 16, 48, 32],
        [96, 48, 104, 8, 24, 32],
        [80, 56, 112, 12, 32, 32],
        [64, 64, 128, 12, 32, 32],
        [56, 72, 144, 16, 32, 32],
        [128, 80, 160, 16, 64, 64],
        [128, 80, 160, 16, 64, 64],
        [192, 96, 208, 16, 48, 64],
    ]

    parameters = np.array(parameters)
    parameters *= 2
    parameters = list(parameters)

    channel_lists = list()
    channel_lists.append(make_channel_list(
        step_1_channel, *parameters[0]))     # 3a 128
    channel_lists.append(make_channel_list(
        get_block_output_size(channel_lists[0]), *parameters[1]))    # 3b 240

    channel_lists.append(make_channel_list(
        get_block_output_size(channel_lists[1]), *parameters[2]))    # 4a 256
    channel_lists.append(make_channel_list(
        get_block_output_size(channel_lists[2]), *parameters[3]))    # 4b 256
    channel_lists.append(make_channel_list(
        get_block_output_size(channel_lists[3]), *parameters[4]))    # 4c 256
    channel_lists.append(make_channel_list(
        get_block_output_size(channel_lists[4]), *parameters[5]))    # 4d 264
    channel_lists.append(make_channel_list(
        get_block_output_size(channel_lists[5]), *parameters[6]))    # 4e 416

    channel_lists.append(make_channel_list(
        get_block_output_size(channel_lists[6]), *parameters[7]))    # 5a 416
    channel_lists.append(make_channel_list(
        get_block_output_size(channel_lists[7]), *parameters[8]))    # 5b 512

    model = GoogLeNet(
        input_channel=3,
        step_1_channel=step_1_channel,
        class_num=10,
        channel_lists=channel_lists
    )

    return model


if __name__ == "__main__":
    import numpy as np
    google_model = my_googLeNet()
    params = list(google_model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
        # print(l)
    k = format(k, ',')
    print("total parameters: " + k)
    #
    x = np.random.random([1, 3, 32, 32])
    x = torch.Tensor(x)
    y, _, _ = google_model(x)
    print(y.size())
