import os
import json
import sys
import subprocess
import tarfile
from pathlib import Path
from tqdm import tqdm
from Config import Config
from datasets import load_dataset


def process_pretrain_data(config):
    """处理预训练数据"""
    print("\n--- 处理预训练数据 ---")
    input_path = Path(config.DATA_DIR) / config.PRETRAIN_FILE
    output_path = Path(config.DATA_DIR) / 'seq_monkey_datawhale.jsonl'

    if not input_path.exists():
        print(f"错误：找不到预训练输入文件: {input_path}")
        print("请确保步骤1已成功执行。")
        sys.exit(1)

    def split_text(text, chunk_size=512):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    try:
        processed_count = 0
        error_count = 0

        with open(output_path, 'w', encoding='utf-8') as pretrain:
            # 使用流式读取，避免一次性加载整个文件
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc="处理预训练数据"), 1):
                    try:
                        line = line.strip()
                        if not line:  # 跳过空行
                            continue

                        line_data = json.loads(line)
                        text = line_data.get('text', '')

                        if not text.strip():  # 跳过空文本
                            continue

                        chunks = split_text(text)
                        for chunk in chunks:
                            if chunk.strip():
                                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
                                processed_count += 1

                        # 每处理1000行刷新一次缓冲区
                        if processed_count % 1000 == 0:
                            pretrain.flush()

                    except json.JSONDecodeError:
                        error_count += 1
                        if error_count <= 10:  # 只显示前10个错误
                            print(f"警告：第{line_num}行JSON解析错误: {line[:50]}...")
                        continue
                    except Exception as e:
                        error_count += 1
                        if error_count <= 10:
                            print(f"警告：第{line_num}行处理错误: {e}")
                        continue

        print(f"\n预训练数据处理完成！")
        print(f"成功处理: {processed_count} 个文本块")
        print(f"错误行数: {error_count}")
        print(f"输出文件: {output_path}")

    except Exception as e:
        print(f"处理预训练数据时发生错误: {e}")
        sys.exit(1)


def process_sft_data(config):
    """处理SFT数据"""
    print("\n--- 处理SFT数据 ---")
    input_path = Path(config.DATA_DIR) / 'Bellegroup' / config.SFT_FILE
    output_path = Path(config.DATA_DIR) / 'BelleGroup_sft.jsonl'

    if not input_path.exists():
        print(f"错误：找不到SFT输入文件: {input_path}")
        print("请确保步骤1已成功执行，并且 Config.py 中的 SFT_FILE 名称正确。")
        sys.exit(1)

    def convert_message(data):
        message = [{"role": "system", "content": "你是一个AI助手"}]
        for item in data:
            if item.get('from') == 'human':
                message.append({'role': 'user', 'content': item.get('value', '')})
            elif item.get('from') == 'assistant':
                message.append({'role': 'assistant', 'content': item.get('value', '')})
        return message

    try:
        processed_count = 0
        error_count = 0

        with open(output_path, 'w', encoding='utf-8') as sft:
            # 使用流式读取，避免一次性加载整个文件
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc="处理SFT数据"), 1):
                    try:
                        line = line.strip()
                        if not line:  # 跳过空行
                            continue

                        item_data = json.loads(line)
                        if 'conversations' not in item_data:
                            error_count += 1
                            if error_count <= 5:  # 只显示前5个警告
                                print(f"警告：第{line_num}行缺少 'conversations' 键")
                            continue

                        # 转换消息格式
                        message = convert_message(item_data['conversations'])
                        sft.write(json.dumps(message, ensure_ascii=False) + '\n')
                        processed_count += 1

                        # 每处理1000行刷新一次缓冲区
                        if processed_count % 1000 == 0:
                            sft.flush()

                    except json.JSONDecodeError as e:
                        error_count += 1
                        if error_count <= 5:
                            print(f"警告：第{line_num}行JSON解析错误: {e}")
                        continue
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:
                            print(f"警告：第{line_num}行处理错误: {e}")
                        continue

        print(f"\nSFT数据处理完成！")
        print(f"成功处理: {processed_count} 条对话")
        print(f"错误行数: {error_count}")
        print(f"输出文件: {output_path}")

    except Exception as e:
        print(f"处理SFT数据时发生错误: {e}")
        sys.exit(1)


def main(config):


    # 1. 处理预训练数据
    process_pretrain_data(config)

    # 2. 处理SFT数据
    #process_sft_data(config)

    print("\n数据处理全部完成！")

if __name__ == "__main__":
    try:
        main(Config)
    except Exception as e:
        print(f"脚本执行时发生意外的顶层错误: {e}")
        sys.exit(1)