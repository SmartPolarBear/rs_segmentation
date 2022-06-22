import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager

class MultiClassFocalLoss(nn.Layer):
    """
    The implement of focal loss for multi class.
    Args:
        alpha (float, list, optional): The alpha of focal loss. alpha is the weight
            of class 1, 1-alpha is the weight of class 0. Default: 0.25
        gamma (float, optional): The gamma of Focal Loss. Default: 2.0
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, num_class, alpha=1.0, gamma=2.0, ignore_index=255):
        super().__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.EPS = 1e-10

    def forward(self, logit, label):
        """
        Forward computation.
        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C, H, W), where C is number of classes.
            label (Tensor): Label tensor, the data type is int64. Shape is (N, W, W),
                where each value is 0 <= label[i] <= C-1.
        Returns:
            (Tensor): The average loss.
        """
        assert logit.ndim == 4, "The ndim of logit should be 4."
        assert label.ndim == 3, "The ndim of label should be 3."

        logit = paddle.transpose(logit, [0, 2, 3, 1])
        label = label.astype('int64')
        ce_loss = F.cross_entropy(
            logit, label, ignore_index=self.ignore_index, reduction='none')

        pt = paddle.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt)**self.gamma) * ce_loss

        mask = paddle.cast(label != self.ignore_index, 'float32')
        focal_loss *= mask
        avg_loss = paddle.mean(focal_loss) / (paddle.mean(mask) + self.EPS)
        return avg_loss