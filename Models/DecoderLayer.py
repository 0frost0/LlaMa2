from torch import nn

from Config import ModelConfig
from Models import Attention, MLP
from Models.RMSNorm import RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # 定义多头注意⼒的头数
        self.n_heads = args.n_heads
        # 定义输⼊维度
        self.dim = args.dim
        # 定义每个头的维度，等于输⼊维度除以头数
        self.head_dim = args.dim // args.n_heads
        # 定义LLaMA2Attention对象，⽤于进⾏多头注意⼒计算
        self.attention = Attention(args)
        # 定义LLaMAMLP对象，⽤于进⾏前馈神经⽹络计算
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # 定义层的ID
        self.layer_id = layer_id
        # 定义注意⼒计算的归⼀化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经⽹络计算的归⼀化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):

        # 前向传播函数
        # ⾸先，输⼊x经过注意⼒归⼀化层，然后进⾏注意⼒计算，结果与输⼊x相加得到h
        # 然后， h经过前馈神经⽹络归⼀化层，然后进⾏前馈神经⽹络计算，结果与h相加得到输出
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out