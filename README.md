# LLaMA2-Tiny Implementation from Scratch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./LICENSE)

本项目是基于 [Datawhale Happy-LLM (从零开始的大模型)](https://github.com/datawhalechina/happy-llm) 教程第五章内容的复现与实现。

项目旨在**从零开始（From Scratch）**构建并训练一个基于 LLaMA2 架构的小型语言模型（Tiny-LLM）。项目包含完整的模型结构实现（RMSNorm, RoPE, GQA, SwiGLU）、Tokenizer 训练、预训练（Pretrain）以及有监督微调（SFT）的全流程代码。

## 📖 项目特色

* **模型架构**：完全手写实现 LLaMA2 核心组件，包括：
    * **RMSNorm** (Root Mean Square Layer Normalization)
    * **RoPE** (Rotary Positional Embeddings) 旋转位置编码
    * **GQA** (Grouped-Query Attention) 分组查询注意力机制
    * **SwiGLU** 激活函数的前馈神经网络
* **全流程训练**：包含从数据清洗、Tokenizer 训练、模型预训练到指令微调（SFT）的完整流水线。
* **分布式训练**：支持使用 **DeepSpeed** 进行多卡分布式训练。
* **实验追踪**：集成 **SwanLab** 进行训练过程的可视化监控。

## ⚙️ 模型配置 (Tiny-LLM)

本项目默认训练的模型配置如下（约 215M 参数）：

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| `dim` | 1024 | 隐藏层维度 |
| `n_layers` | 18 | Transformer 层数 |
| `n_heads` | 16 | 注意力头数 |
| `n_kv_heads` | 8 | KV 头数 (GQA) |
| `vocab_size` | 6144 | 词表大小 |
| `max_seq_len` | 512 | 最大序列长度 |

## 🛠️ 环境安装

1. 克隆仓库：
   ```bash
   git clone [https://github.com/0frost0/LlaMa2.git](https://github.com/0frost0/LlaMa2.git)
   cd LlaMa2
