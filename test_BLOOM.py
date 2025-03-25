import pandas as pd
from transformers import pipeline, BloomTokenizerFast
import chardet
import torch
from tqdm import tqdm

count = 0
# 配置输入输出文件
INPUT_CSV = 'querys.csv'
OUTPUT_CSV = 'test_bloom_new.csv'

# 模型加载 (保持原结构)
device = "cuda" if torch.cuda.is_available() else "cpu"
bloom_generator = pipeline("text-generation", model="bigscience/bloom-560m", device=device)
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")


def add_system_prompt(text):
    """优化模板"""
    return f"""请根据以下问题提供专业的中文回答。
问题：{text}
回答："""


def generate_content_bloom(text):
    """优化后的生成函数"""
    prompted_text = add_system_prompt(text)
    input_length = len(tokenizer.encode(prompted_text))
    max_length = min(input_length + 400, 1024)  # 更安全的长度限制

    global count
    print(count,end="")
    count = count + 1

    result = bloom_generator(
        prompted_text,
        max_length=max_length,
        num_return_sequences=1,
        truncation=True,
        do_sample=True,  # 启用采样
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.2
    )

    # 提取回答部分
    full_text = result[0]['generated_text']
    answer = full_text.split("回答：")[-1].strip()
    return answer.replace('\n', ' ')


# 文件处理 (保持原结构)
with open(INPUT_CSV, 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']

df = pd.read_csv(INPUT_CSV, encoding=encoding)

# 检查列名 (适配单query列)
if 'query' not in df.columns:
    raise ValueError("CSV文件必须包含 'query' 列")

# 使用进度条
df['generated_content'] = [generate_content_bloom(q) for q in tqdm(df['query'], desc="Generating")]

# 保存到新文件 (不再覆盖原文件)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"处理完成，结果已保存到: {OUTPUT_CSV}")