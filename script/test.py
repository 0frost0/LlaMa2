from Models import Attention,precompute_freqs_cis,MLP
import torch
from torch import nn
from Config import ModelConfig

def test_attention(x):
    """测试Attention模型的简单函数"""
    # 创建Attention实例
    attention_model = Attention(args)

    # 预计算RoPE频率
    print("预计算RoPE频率")
    freqs_cos, freqs_sin = precompute_freqs_cis(dim // args.n_heads, seq_len)

    # 运行Attention模型
    print("运行Attention模型")
    output = attention_model(x, freqs_cos, freqs_sin)

    # 检查输出形状 应该是[batch_size, seq_len, dim]
    print("Output shape:", output.shape)

    # 验证形状是否正确
    expected_shape = (batch_size, seq_len, dim)
    assert output.shape == expected_shape, f"形状不匹配: 期望{expected_shape}, 实际{output.shape}"

    print("✅ Attention测试通过!")
    return output
def test_MLP(x):
    mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
    # 随机⽣成数据
    #x = torch.randn(1, 50, args.dim)
    # 运⾏MLP模型
    output = mlp(x)
    print("Output shape:", output.shape)
    print("✅ MLP测试通过!")
    return output

# 运行测试
# 先实例化args
print("实例化args")
args = ModelConfig(
    dim=768,
    n_layers=12,
    n_heads=16,
    n_kv_heads=8,
    vocab_size=6144,
    max_seq_len=512
)
# 模拟输入数据
print("模拟输入数据")
batch_size = 1
seq_len = 50  # 假设实际使用的序列长度为50
dim = args.dim
x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量

output1 = test_attention(x)
output2 = test_MLP(output1)

