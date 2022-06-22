
import paddle
import paddle.nn as nn

class ChannelAttention(nn.Layer):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2D(in_planes, in_planes // rotio , 1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(in_planes // rotio, in_planes, 1, bias_attr=False))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"

        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2D(2,1,kernel_size, padding=padding, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = paddle.mean(x, axis=1, keepdim=True)
        maxout = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avgout, maxout], axis=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Layer):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels,rotio=reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x  
        x = self.sa(x) * x 

        return x