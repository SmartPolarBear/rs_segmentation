import paddle
import paddle.nn as nn

class SEBlock(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape([b, c]) # squeeze操作
        y = self.fc(y).reshape([b, c, 1, 1]) # FC获取通道注意力权重，是具有全局信息的
        return x * y # 注意力作用每一个通道上