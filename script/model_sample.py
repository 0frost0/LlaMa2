import os
import pickle
from contextlib import nullcontext
import torch
from Config import ModelConfig,Config
from Models import  Transformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

class TextGenerator:
    def __init__(self,
                 checkpoint='./base_model_215M/pretrain_1024_18_6144.pth',  # 模型检查点路径
                 tokenizer_model_path=Config.TOKENIZER_FILE,  # 分词器模型路径
                 seed=42,  # 随机种⼦，确保可重复性
                 device=None,  # 设备，优先使⽤ CUDA，如果没有可⽤的 CUDA，则使⽤ CPU
                 dtype="bfloat16"):  # 数据类型，默认为 float32，可以选择 float16 或 bfloat16
        """
        初始化 TextGenerator 类，加载模型、设置设备和分词器等。
        """
        # 模型加载配置
        self.checkpoint = checkpoint  # 保存的模型检查点路径
        self.tokenizer_model_path = tokenizer_model_path  # 分词器模型⽂件路径
        self.seed = seed  # 随机数种⼦，⽤于⽣成的可重复性
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')  # 根据硬件条件选择设备
        self.dtype = dtype  # 模型的浮点数类型
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'  # 判断当前设备是否为CUDA
        # 设置随机种⼦，确保⽣成的可重复性
        torch.manual_seed(seed)  # 设置 CPU 随机种⼦
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种⼦
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许 CUDA 使⽤ TF32 精度进⾏矩阵乘法运算
        torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使⽤ TF32 精度加速
        # 根据 dtype 选择适当的⾃动混合精度上下⽂
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16':torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        # 加载模型检查点⽂件
        checkpoint_dict = torch.load(self.checkpoint, map_location=self.device)  # 加载模型参数  # 初始化模型参数
        self.model = Transformer(ModelConfig(dim=1024, n_layers=18))  # 实例化 Transformer模型
        sunwanted_prefix = '_orig_mod.'
        for k, v in list(checkpoint_dict.items()):
            if k.startswith(sunwanted_prefix):
                checkpoint_dict[k[len(sunwanted_prefix):]] = checkpoint_dict.pop(k)
        self.model.load_state_dict(checkpoint_dict, strict=False)
        # 计算模型参数量
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {num_params / 1e6:.3f} M parameters.")
        # 设置模型为评估模式（evaluation mode），防⽌训练模式下的 dropout 等操作影响结果
        self.model.eval()
        # 将模型放置到正确的设备上（GPU 或 CPU）
        self.model.to(self.device)
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_path)  # 根据指定的路径加载分词器

    def chat_template(self, prompt):
        message = [
            {"role": "system", "content": "你是⼀个AI助⼿，你的名字叫⼩明。 "},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False,
                                                  add_generation_prompt=True)

    def sft_sample(self,
                   start="Hello!",  # ⽣成⽂本的起始提示词，可以是任意字符串
                   num_samples=3,  # ⽣成样本的数量，默认⽣成 3 个样本
                   max_new_tokens=256,  # 每个样本⽣成的最⼤ token 数，默认最多⽣成 256 个 token
                   temperature=0.7,  # 控制⽣成的随机性， 1.0 为标准，值越⼤越随机
                   top_k=300):  # 保留概率最⾼的 top_k 个 token，限制⽣成时的选择范围
        """
        :param start:⽣成⽂本的起始提示词
        :param num_samples:要⽣成的⽂本样本数
        :param max_new_tokens: 每个样本⽣成的最⼤ token 数
        :param temperature:控制⽣成的随机性，值越⼩⽣成越确定，值越⼤⽣成越随机
        :param top_k: 限制⽣成时选择的 token 范围
        :return:⽣成的⽂本样本列表
        """
        start = self.chat_template(start)
        # 将起始⽂本编码为 token id 序列
        start_ids = self.tokenizer(start).data['input_ids']
        # print('start_ids:', start_ids)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])  #
        generated_texts = []  # ⽤于保存⽣成的⽂本样本
        with torch.no_grad():  # 禁⽤梯度计算，提升效率
            with self.ctx:  # 进⼊⾃动混合精度的上下⽂（如果是 GPU 并使⽤ float16 时）
                for k in range(num_samples):  # 循环⽣成指定数量的样本
                    y = self.model.generate(x, self.tokenizer.eos_token_id,max_new_tokens, temperature=temperature, top_k=top_k) # ⽣成⽂本
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()))
        return generated_texts  # 返回⽣成的⽂本样本

    def pretrain_sample(self,
                        start="Hello!",  # ⽣成⽂本的起始提示词，可以是任意字符串
                        num_samples=3,  # ⽣成样本的数量，默认⽣成 3 个样本
                        max_new_tokens=256,  # 每个样本⽣成的最⼤ token 数，默认最多⽣成 256 个 token
                        temperature=0.7,  # 控制⽣成的随机性， 1.0 为标准，值越⼤越随机
                        top_k=300):  # 保留概率最⾼的 top_k 个 token，限制⽣成时的选择范围
        """
        :param start:
        :param num_samples:
        :param max_new_tokens:
        :param temperature:
        :param top_k:
        :return:
        """
        # 如果 start 是以 'FILE:' 开头，表示从⽂件中读取起始⽂本
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()  # 读取⽂件内容作为起始⽂本
        # 将起始⽂本编码为 token id 序列
        start_ids = self.tokenizer(start).data['input_ids']
        # print('start_ids:', start_ids)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])
        generated_texts = []
        with torch.no_grad():  # 禁⽤梯度计算，提升效率
            with self.ctx:  # 进⼊⾃动混合精度的上下⽂（如果是 GPU 并使⽤ float16 时）
                for k in range(num_samples):  # 循环⽣成指定数量的样本
                    y = self.model.generate(x, max_new_tokens=max_new_tokens,temperature=temperature, top_k=top_k)  # ⽣成⽂本
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()))  # 解码⽣

        return generated_texts  # 返回⽣成的⽂本样本

if __name__ == "__main__":
    print("------------------- Pretrain Sample ------------------- \n")
    pretrain_prompt_datas = [
        '<|im_start|>清华大学是',
        '<|im_start|>中国科学院大学深圳先进技术研究院',
    ]
    generator = TextGenerator(checkpoint='./base_model_215M/pretrain_1024_18_6144.pth')  #初始化⽣成器
    for i in range(len(pretrain_prompt_datas)):
        samples = generator.pretrain_sample(start=pretrain_prompt_datas[i], num_samples=1,
                                            max_new_tokens=120, temperature=0.75)
        print(f"\nSample {i + 1}:\n{pretrain_prompt_datas[i]}{samples[0]}\n{'-' * 20}")  # 打印⽣成的样本并⽤分隔线分割
    print("\n ------------------- SFT Sample ------------------- \n")
    sft_prompt_datas = [
        '你好呀，我叫卡夫卡',
        "中国的⾸都是哪⾥？ ",
        "1+1等于多少？ ",
        "我叫什么？ ",
    ]
    generator = TextGenerator(checkpoint='./sft_model_215M/sft_dim1024_layers18_vocab_size6144.pth')  # 初始化⽣成器
    for i in range(len(sft_prompt_datas)):
        samples = generator.sft_sample(start=sft_prompt_datas[i], num_samples=1,
                                       max_new_tokens=128, temperature=0.6)
        print(f"\nSample {i + 1}:\nQuestion: {sft_prompt_datas[i]} \nAI answer:{samples[0]}\n{'-' * 20}") # 打印⽣成的样本并⽤分隔线分割




