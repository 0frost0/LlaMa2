from transformers import PretrainedConfig
import torch

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"

    def __init__(
            self,
            dim: int = 768,  # 模型维度
            n_layers: int = 12,  # Transformer的层数
            n_heads: int = 16,  # 注意⼒机制的头数
            n_kv_heads: int = 8,  # 键值头的数量
            vocab_size: int = 6144,  # 词汇表⼤⼩
            hidden_dim: int = None,  # 隐藏层维度
            multiple_of: int = 64,
            norm_eps: float = 1e-5,  # 归⼀化层的eps
            max_seq_len: int = 512,  # 最⼤序列⻓度
            dropout: float = 0.0,  # dropout概率
            flash_attn: bool = True,  # 是否使⽤Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
class Config:
    DATA_DIR = "/root/autodl-tmp/LAY_workspace/LlaMa2/"
    PRETRAIN_FILE = 'mobvoi_seq_monkey_general_open_corpus.jsonl'
    SFT_FILE = 'train_3.5M_CN.json'
    TOKENIZER_FILE = '/home/LAY/LlaMa2/tokenizer_k/'