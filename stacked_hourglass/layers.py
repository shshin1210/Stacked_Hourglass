import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, bn=False, relu=False): # 3x3 conv layer
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)  # padding = kernel_size - 1
        self.relu = None
        self.bn = None

        if relu: # relu가 true이면 nn.ReLU() 정의하기
            self.relu = nn.ReLU()
        if bn: # bach normalization이 true이면 nn.BatchNorm2d() 정의하기
            self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x): # x를 conv layer에 넣어주기/ bn 과 relu true false 에 따라 넣어주기
       x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual, self).__init__()
        self.skip_layer = Conv(in_channel, out_channel, 1, relu=False) # dimension 맞춰주는용 1x1 conv layer

        if in_channel == out_channel: # input channel과 output channel이 같을 때, Skip connection 불필요
            self.need_skip = False
        else:
            self.need_skip = True

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv1 = Conv(in_channel, out_channel // 2, 1, bn=True, relu=True) # out_channel // 2 = out_channel을 2 로 나누었을때의 몫
        self.conv2 = Conv(out_channel // 2, out_channel // 2, 3, bn=True, relu=True)
        self.conv3 = Conv(out_channel // 2, out_channel, 1)

    def forward(self, x):
        if self.need_skip: #need_skip이 true 일 때 skip layer 적용
            residual = self.skip_layer(x)
        else:
            residual = x # 그냥 x 가져옴
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return residual + out # skip connection이랑 더해주는 역할


class Hourglass(nn.Module:)
    def __init__(self, layer, channel, inc=0):
        super(Hourglass, self).__init__()
        nf = channel + inc #inc : increase
        self.res = Residual(channel, channel)
        self.pool = nn.MaxPool2d(2, 2)
        self.res1 = Residual(channel, nf) #nf = channel + inc

        if layer > 1:  #layer = 4
            self.hourclass = Hourglass(layer - 1, nf) # 재귀로 다시 class에 대입, layer =1이 될때까지
        else:
            self.hourclass = Residual(nf, nf)  # 마지막 hourclass = residual

        self.res2 = Residual(nf, channel)
        self.up = nn.Upsample(scale_factor=2, mode='nearest') # upsampling : 데이터의 크기 키우기

    def forward(self, x):
        res = self.res(x) # 추가로 더해줄 x
        x = self.pool(x)
        x = self.res1(x)
        x = self.hourclass(x)
        x = self.res2(x)
        x = self.up(x)
        return res + x


if __name__ == '__main__':
    model = Hourglass(2, 8)
    x = torch.randn((1, 8, 16, 16))
    out = model(x)
    print(out.shape)
