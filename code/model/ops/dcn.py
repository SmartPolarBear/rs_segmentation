import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.vision.ops import DeformConv2D

class DeformableConvV2(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 regularizer=None,
                 lr_scale=1.,
                 skip_quant=False,
                 dcn_bias_regularizer=paddle.regularizer.L2Decay(0.),
                 dcn_bias_lr_scale=2.,
                 data_format="NCHW"):
        super().__init__()
        self.offset_channel = 2 * kernel_size**2
        self.mask_channel = kernel_size**2

        offset_bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0.),
            learning_rate=lr_scale,
            regularizer=regularizer)

        self.conv_offset = nn.Conv2D(
            in_channels,
            3 * kernel_size**2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0)),
            bias_attr=offset_bias_attr,
            data_format=data_format)

        if bias_attr:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            dcn_bias_attr = paddle.ParamAttr(
                initializer=nn.initializer.Constant(value=0),
                regularizer=dcn_bias_regularizer,
                learning_rate=dcn_bias_lr_scale)
        else:
            # in ResNet backbone, do not need bias
            dcn_bias_attr = False
        self.conv_dcn = DeformConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=dcn_bias_attr)

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y