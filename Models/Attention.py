import torch
from torch import nn
from typing import Tuple, Optional
import math
import torch.nn.functional as F
from Config import ModelConfig

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取输⼊张量的形状：批量⼤⼩、序列⻓度、键/值对头的数量、每个头的维度⼤⼩
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 对张量进⾏扩展和重塑操作以重复键值对
    return (
        x[:, :, :, None, :]  # 在第四个维度（头的维度前）添加⼀个新的维度
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # 将新添加的维度扩展到n_rep⼤⼩，实现重复的效果
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # 重新塑形，合并键/值对头的数量和重复次数的维度
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()⽣成了⼀个从0开始，步⻓为2的序列，⻓度为dim的⼀半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # ⽣成⼀个从0到end的序列，⻓度为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到⼀个⼆维矩阵，每⼀⾏是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取x的维度数
    ndim = x.ndim
    # 断⾔，确保1在x的维度范围内
    assert 0 <= 1 < ndim
    # 断⾔，确保freqs_cis的形状与x的第⼆维和最后⼀维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # 构造⼀个新的形状，除了第⼆维和最后⼀维，其他维度都为1，这样做是为了能够将freqs_cis与x进⾏⼴播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    # 重新塑形频率张量以进⾏⼴播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应⽤旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # 根据是否指定n_kv_heads，确定⽤于键（key）和值（value）的头的数量。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数可以被键值头数整除。
        assert args.n_heads % self.n_kv_heads == 0
        # 模型并⾏处理⼤⼩，默认为1。
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并⾏处理⼤⼩。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值头数除以模型并⾏处理⼤⼩。
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，⽤于扩展键和值的尺⼨。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads
        # 定义权重矩阵。
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出权重矩阵。
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # 定义dropout。
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存dropout概率。
        self.dropout = args.dropout
        # 检查是否使⽤Flash Attention（需要PyTorch >= 2.0）。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 若不⽀持Flash Attention，则使⽤⼿动实现的注意⼒机制，并设置mask。
            print("WARNING: using slow attention. Flash Attention requires PyTorch >=2.0")
            # 创建⼀个上三⻆矩阵，⽤于遮蔽未来信息。
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):

        # 获取批次⼤⼩和序列⻓度， [batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状以适应头的维度。
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # 应⽤旋转位置嵌⼊（RoPE）。
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        # 对键和值进⾏扩展以适应重复次数。
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)
        # 将头作为批次维度处理。
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # 根据是否⽀持Flash Attention，选择实现⽅式。
        if self.flash:
            # 使⽤Flash Attention。
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv,
                                                                      attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            # 使⽤⼿动实现的注意⼒机制。
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)
        # 恢复时间维度并合并头。
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output