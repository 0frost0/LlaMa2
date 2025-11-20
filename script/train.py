import argparse
import os
import time
from logging import Logger
import swanlab
from contextlib import nullcontext

from torch import optim
from torch.utils.data import DataLoader

import math
import torch
from transformers import AutoTokenizer

from Config import ModelConfig,Config
from Models import Transformer
from data import PretrainDataset


def get_lr(it, all):
    """
    计算当前迭代的学习率，使⽤余弦退⽕调度策略
    学习率调度策略：
    1. Warmup阶段：学习率从0线性增⻓到⽬标学习率
    2. 余弦退⽕阶段：学习率按余弦函数衰减到最⼩学习率
    3. 超出训练步数后：保持最⼩学习率Args:
    it (int): 当前迭代步数
    all (int): 总迭代步数
    Returns:
    float: 当前步数对应的学习率
    """
    warmup_iters = args.warmup_iters  # 预热迭代次数
    lr_decay_iters = all  # 学习率衰减的总迭代次数
    min_lr = args.learning_rate / 10  # 最⼩学习率，为初始学习率的1/10
    # Warmup阶段：线性增⻓
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    # 超出训练步数：保持最⼩学习率
    if it > lr_decay_iters:
        return min_lr
    # 余弦退⽕阶段
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 余弦系数
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """
    训练⼀个epoch的函数
    实现了完整的训练循环，包括：
    1. 数据加载和设备转移
    2. 动态学习率调整
    3. 前向传播和损失计算
    4. 梯度累积和反向传播
    5. 梯度裁剪和优化器更新
    6. ⽇志记录和模型保存
    Args:
    epoch (int): 当前epoch编号
    """
    start_time = time.time() # 记录开始时间
    # 遍历数据加载器中的每个batch
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据转移到指定设备（GPU/CPU）
        X = X.to(args.device)  # 输⼊序列
        Y = Y.to(args.device)  # ⽬标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码，⽤于忽略padding token
        # 计算当前步骤的学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使⽤混合精度训练上下⽂
        with ctx:
            # 前向传播
            out = model(X, Y)
            # 计算损失并除以累积步数（⽤于梯度累积）
            loss = out.last_loss / args.accumulation_steps
            # 将loss_mask展平为⼀维
            loss_mask = loss_mask.view(-1)
            # 应⽤掩码计算有效损失（忽略padding位置）
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # 使⽤scaler进⾏混合精度的反向传播
        scaler.scale(loss).backward()
        # 每accumulation_steps步执⾏⼀次优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放，准备梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪，防⽌梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 执⾏优化器步骤
            scaler.step(optimizer)
            # 更新scaler的缩放因⼦
            scaler.update()
            # 清零梯度， set_to_none=True可以节省内存
            optimizer.zero_grad(set_to_none=True)
        # 每log_interval步记录⼀次⽇志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 打印训练进度信息
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复真实的loss值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            # 如果启⽤SwanLab，记录训练指标
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })
        # 每save_interval步保存⼀次模型
        if (step + 1) % args.save_interval == 0:
            model.eval()  # 切换到评估模式
            # 构建检查点⽂件名
            ckp =f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'
            # 处理多卡保存：如果是DataParallel模型，需要访问.module属性
            state_dict = model.module.state_dict() if isinstance(model,torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()  # 切换回训练模式
            # 每20000步保存⼀个带步数标记的检查点
            if (step + 1) % 20000 == 0:
                model.eval()
            # 构建带步数的检查点⽂件名
            ckp =f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step + 1}.pth'
            # 保存模型状态字典
            state_dict = model.module.state_dict() if isinstance(model,torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()
def init_model():
    """
    初始化模型和分词器
    功能包括：
    1. 加载预训练的分词器
    2. 创建Transformer模型
    3. 设置多GPU并⾏训练（如果可⽤）
    4. 将模型移动到指定设备
    5. 统计并打印模型参数量
    Returns:
    tuple: (model, tokenizer) 初始化后的模型和分词器
    """
    def count_parameters(model):
        """
        统计模型中可训练参数的数量
        Args:
        model: PyTorch模型
        Returns:
            int: 可训练参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 从本地路径加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained('/home/LAY/LlaMa2/tokenizer_k/')
    print("\n---加载分词器完成完成---")
    # 根据配置创建Transformer模型
    model = Transformer(lm_config)
    #加载已训练的模型
    ckp_path = f'{args.out_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'
    if os.path.exists(ckp_path):
        print(f"\n--- 发现已存在的检查点: {ckp_path} ---")
        print("--- 正在加载模型权重... ---")
        try:
            # 3. 加载检查点
            model.load_state_dict(torch.load(ckp_path, map_location=args.device))
            print("--- 模型权重加载成功！将从断点处继续训练 ---")
        except Exception as e:
            print(f"!! 加载模型失败: {e} !!")
            print("!! 将从头开始训练 !!")
    else:
        print(f"\n--- 未发现检查点, 将从头开始训练 ---")
    print("\n---配置Transformer完成---")
    # 多卡初始化：检查可⽤GPU数量并设置DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        # 使⽤DataParallel包装模型以⽀持多GPU训练
        model = torch.nn.DataParallel(model)
    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(args.device)
    # 计算并打印模型参数量（以百万为单位）
    Logger(f'LLM总参数量： {count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == '__main__':
    # ==================== 命令⾏参数解析 ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    # 基础训练参数
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="模型输出⽬录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=48, help="批次⼤⼩")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if
    torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    # 实验跟踪和数据加载参数
    parser.add_argument("--use_swanlab", action="store_false", help="是否使⽤SwanLab进⾏实验跟踪")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的⼯作进程数")
    parser.add_argument("--data_path", type=str, default=os.path.join(Config.DATA_DIR, "seq_monkey_datawhale.jsonl"),help="训练数据路径")
    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代次数")
    # ⽇志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="⽇志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    # 多GPU训练参数
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="使⽤的GPU ID，⽤逗号分隔(例如: '0,1,2')")
    args = parser.parse_args()
    print("\n---参数配置完成---")
    # ==================== GPU环境设置 ====================
    # 设置可⻅的GPU设备
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # ⾃动设置主设备为第⼀个可⽤GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"

        else:
            args.device = "cpu"
    print(f"\n---{args.device}---")
    # ==================== 实验跟踪初始化 ====================
    if args.use_swanlab:
        # 注意：使⽤前需要先登录 swanlab.login(api_key='your key')
        run = swanlab.init(
            project="LlaMa2",  # 项⽬名称
            experiment_name="Pretrain-215M",  # 实验名称
            config=args,  # 保存所有超参数
        )
    # ==================== 模型配置 ====================
    # 定义语⾔模型的配置参数
    lm_config = ModelConfig(
        dim=1024,  # 模型维度
        n_layers=18,  # Transformer层数
    )
    # ==================== 训练环境设置 ====================
    max_seq_len = lm_config.max_seq_len  # 最⼤序列⻓度
    args.save_dir = os.path.join(args.out_dir)  # 模型保存⽬录
    # 创建必要的⽬录
    os.makedirs(args.out_dir, exist_ok=True)
    # 设置随机种⼦以确保结果可复现
    torch.manual_seed(42)
    # 确定设备类型（⽤于选择合适的上下⽂管理器）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 设置混合精度训练的上下⽂管理器
    # CPU训练时使⽤nullcontext， GPU训练时使⽤autocast
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda')
    print("\n---训练环境设置完成---")
    # ==================== 模型和数据初始化 ====================
    # 初始化模型和分词器
    model, tokenizer = init_model()
    # 创建训练数据集
    print("\n---创建训练数据集完成---")
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,  # 批次⼤⼩
        pin_memory=True,  # 将数据加载到固定内存中，加速GPU传输
        drop_last=False,  # 不丢弃最后⼀个不完整的批次
        shuffle=True,  # 随机打乱数据
        num_workers=args.num_workers  # 数据加载的并⾏⼯作进程数
    )
    print("\n---创建数据加载器完成---")
    # ==================== 优化器和训练组件初始化 ====================
    # 初始化混合精度训练的梯度缩放器
    # 只有在使⽤float16或bfloat16时才启⽤
    scaler = torch.amp.GradScaler('cuda',enabled=(args.dtype in ['float16', 'bfloat16']))
    # 初始化Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # ==================== 开始训练 ====================
    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    # 开始训练循环
    print("\n---开始循环训练---")
    for epoch in range(args.epochs):
        train_epoch(epoch)


