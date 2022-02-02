import torch
import torch.nn as nn
from layers import Conv, Hourglass, Residual


class Convert(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convert, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1) # 1x1 conv으로 dim 맞추기

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module): # nstack = 8개의 hourglass 사용, layer 4개, in_channel = 256, out_channel = 16, inc = 0
    def __init__(self, nstack=8, layer=4, in_channel=256, out_channel=16, increase=0):
        super(PoseNet, self).__init__()
        self.nstack = nstack
        self.pre = nn.Sequential(   # initial processing of image
            Conv(3, 64, 7, 2, bn=True, relu=True), # in channel : 3 (RGB), out channel : 64, 7x7, stride = 2 //  w&h size : 256 -> 128
            Residual(64, 128), # in channel : 64, out channel : 128 // w&h size : 128
            nn.MaxPool2d(2, 2), # w&h size : 128 -> 64
            Residual(128, 128), # in channel : 128, out channel : 128 // 2 subsequent residual modules
            Residual(128, in_channel)  # in channel : 128, out channel : 256
        )
        self.hourglass = nn.ModuleList([nn.Sequential(Hourglass(layer, in_channel, inc=increase)) for _ in range(nstack)]) # 8개의 Hourglass 생성
        self.feature = nn.ModuleList([nn.Sequential(Residual(in_channel, in_channel), Conv(in_channel, in_channel, 1, bn=True, relu=True)) for _ in range(nstack)])
        self.outs = nn.ModuleList([Conv(in_channel, out_channel, 1, bn=False, relu=False) for _ in range(nstack)])
        self.merge_feature = nn.ModuleList([Convert(in_channel, in_channel) for _ in range(nstack - 1)]) # 7번동안 convert
        self.merge_pred = nn.ModuleList([Convert(out_channel, in_channel) for _ in range(nstack - 1)]) #7번동안 convert

    def forward(self, x):
        x = self.pre(x) # initial processing of image
        heat_maps = []
        for i in range(self.nstack): # 8개의 stacked hourglass
            hg = self.hourglass[i](x) # i번째 hourglass에 x input
            feature = self.feature[i](hg) # i번째 feature module list에 위 hg 값 대입
            pred = self.outs[i](feature) # pred변수에 i 번째 outs modulelist 에 위 feature 값 대입
            heat_maps.append(pred) # heat_map list에 pred 값 append
            if i < self.nstack - 1: # i 가 7보다 작으면,
                x = x + self.merge_pred[i](pred) + self.merge_feature[i](feature) # feature랑 pred 합치기
        return heat_maps


if __name__ == '__main__':
    model = PoseNet(4, 2, 256, 15)
    x = torch.zeros((1, 3, 256, 256))
    out = model(x)
    for i in range(len(out)):
        print(out[i].shape)
