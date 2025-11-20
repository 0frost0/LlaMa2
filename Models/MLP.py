from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输⼊维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # 定义第⼀层线性变换，从输⼊维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第⼆层线性变换，从隐藏维度到输⼊维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从输⼊维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层，⽤于防⽌过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # 前向传播函数
        # ⾸先，输⼊x通过第⼀层线性变换和SILU激活函数
        # 然后，结果乘以输⼊x通过第三层线性变换的结果
        # 最后，通过第⼆层线性变换和dropout层
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))