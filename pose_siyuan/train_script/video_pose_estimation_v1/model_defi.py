import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import os
class Conv_bn_33(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Conv_bn_33, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(oup, momentum=0.05)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
class Conv_bn_11(nn.Module):
    def __init__(self, inp, oup):
        super(Conv_bn_11, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(oup, momentum=0.05)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv_1 = nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False)
        self.bn_1 = nn.BatchNorm2d(inp * expand_ratio, momentum=0.05)
        self.relu_1 = nn.ReLU(inplace=True)

        self.conv_2 = nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio,
                                bias=False)
        self.bn_2 = nn.BatchNorm2d(inp * expand_ratio, momentum=0.05)
        self.relu_2 = nn.ReLU(inplace=True)

        self.conv_3 = nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False)
        self.bn_3 = nn.BatchNorm2d(oup, momentum=0.05)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu_2(out)
        out = self.conv_3(out)
        out = self.bn_3(out)

        if self.use_res_connect:

            return x + out
        else:
            return out

expand_t = 6
# default_ir_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [expand_t, 24, 2, 2],
#             [expand_t, 32, 3, 2],
#             [expand_t, 64, 4, 2],
#             [expand_t, 96, 3, 1],
#             [expand_t, 160, 3, 2],
#             [expand_t, 320, 1, 1],
#         ]
default_ir_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [expand_t, 32, 2, 1],
            [expand_t, 64, 2, 2],
            [expand_t, 96, 2, 1],
        ]

class BaseNet(nn.Module):
    def __init__(self, width_mult=1., interverted_residual_setting = default_ir_setting, init_state_path=None, ):
        super(BaseNet, self).__init__()
        self.interverted_residual_setting = interverted_residual_setting
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult)
        self.conv_bn_33 = Conv_bn_33(4, input_channel, 2)
        features = []
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel

        self.block_0 = features[0]
        self.block_1 = features[1]
        self.block_2 = features[2]
        self.block_3 = features[3]
        self.block_4 = features[4]
        self.block_5 = features[5]
        self.block_6 = features[6]
        # self.block_7 = features[7]
        # self.block_8 = features[8]
        # self.block_9 = features[9]
        # self.block_10 = features[10]
    def forward(self, x):
        x = self.conv_bn_33(x)
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        # x = self.block_7(x)
        # x = self.block_8(x)
        # x = self.block_9(x)
        # x = self.block_10(x)
        return x


class mobileConv(nn.Module):
    def __init__(self,inp,oup,kernel_size,stride,padding,bias):
        super(mobileConv, self).__init__()
        self.conv_1 = nn.Conv2d(inp , inp, kernel_size, stride, padding, groups=inp,
                                bias=False)
        self.conv_2 = nn.Conv2d(inp , oup, 1, 1, 0,bias=False)

        self.bn_1 = nn.BatchNorm2d(inp , momentum=0.05)
        self.relu_1 = nn.ReLU(inplace=True)
        self.bn_2 = nn.BatchNorm2d(oup, momentum=0.05)
        self.relu_2 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu_2(out)
        return out

class mobileConvDropActivate(nn.Module):
    def __init__(self,inp,oup,kernel_size,stride,padding,bias):
        super(mobileConvDropActivate, self).__init__()
        self.conv_1 = nn.Conv2d(inp , inp, kernel_size, stride, padding, groups=inp,
                                bias=False)
        self.conv_2 = nn.Conv2d(inp , oup, 1, 1, 0,bias=False)

        self.bn_1 = nn.BatchNorm2d(inp , momentum=0.05)
        self.relu_1 = nn.ReLU(inplace=True)
        # self.bn_2 = nn.BatchNorm2d(oup, momentum=0.05)
        # self.relu_2 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu_1(out)
        out = self.conv_2(out)
        return out

class Hourglass(nn.Module):
    def __init__(self, n, f, increase=128):
        super(Hourglass, self).__init__()
        nf = f + increase
        Conv = mobileConv
        # (feature_num, feature_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.up1 = Conv(f, f, kernel_size=3, stride=1, padding=1, bias=True)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Conv(f, nf, kernel_size=3, stride=1, padding=1, bias=True)
        # Recursive hourglass
        if n > 1:
            self.low2 = Hourglass(n - 1, nf, increase)
        else:
            self.low2 = Conv(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.low3 = Conv(nf, f, kernel_size=3, stride=1, padding=1, bias=True)
        self.up2  = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2

class PoseMobileHourglass(nn.Module):
    def __init__(self, width_mult, hg_iters, increase, output):
        super(PoseMobileHourglass, self).__init__()
        self.width_mult = width_mult
        self.baseNet = BaseNet(width_mult)
        ir_setting = self.baseNet.interverted_residual_setting
        base_out = int(ir_setting[-1][1] * width_mult)
        # print(base_out)
        self.hg_1 = Hourglass(hg_iters, base_out, int(increase * width_mult))
        self.out_11 = nn.Conv2d(base_out, output, 1, 1, 0)
        self.out_conv = nn.Conv2d(output, output, 3, 1, 1)

        self.zyh_conv = nn.Conv2d(output, 14, 3, 1, 1)
        self.zyh_up  = nn.Upsample(scale_factor=4)


    def forward(self, x):
        x = self.baseNet(x)
        # print(x.size())
        x = self.hg_1(x)
        x = self.out_11(x)
        x = self.out_conv(x)
        x = self.zyh_conv(x)
        x = self.zyh_up(x)
        return x

def get_model():
    model = PoseMobileHourglass(width_mult=1, hg_iters=4, increase=32, output=45)
    return model


if __name__ == '__main__':
    p = PoseMobileHourglass(width_mult=1, hg_iters=4, increase=32, output=45).cuda()
    v = torch.Tensor(1, 4, int(384 * 1.5), int(384 * 1.5))
    v = torch.autograd.Variable(v).cuda()
    print(v.size())
    out = p(v)
    print(out.size())
    torch.save(p.state_dict(), 'test')