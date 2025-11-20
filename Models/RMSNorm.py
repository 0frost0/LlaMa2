import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()

        # eps是为了防⽌除以0的情况
        self.eps = eps
        # weight是⼀个可学习的参数，全部初始化为1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):

        # 计算RMSNorm的核⼼部分
        # x.pow(2).mean(-1, keepdim=True)计算了输⼊x的平⽅的均值
        # torch.rsqrt是平⽅根的倒数，这样就得到了RMSNorm的分⺟部分，再加上eps防⽌分⺟为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):

        # forward函数是模型的前向传播
        # ⾸先将输⼊x转为float类型，然后进⾏RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的⼀个可学习的缩放因⼦
        output = self._norm(x.float()).type_as(x)
        return output * self.weight