import pandas as pd
from transformers import pipeline, AutoTokenizer
import chardet
import torch

# 加载并配置模型和tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "EleutherAI/gpt-neo-2.7B"

# 初始化模型和tokenizer（确保只加载一次）
gpt_neo_generator = None
tokenizer = None

def init_model():
    global gpt_neo_generator, tokenizer
    if gpt_neo_generator is None:
        gpt_neo_generator = pipeline("text-generation", model=model_name, device=device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

init_model()

# 定义生成内容的函数
def generate_content_gpt_neo(text):
    try:
        # 动态计算 max_length
        input_length = len(tokenizer.encode(text))
        max_length = input_length + 500  # 增加生成长度
        result = gpt_neo_generator(
            text,
            max_length=max_length,
            num_return_sequences=1,
            truncation=True,
        )
        generated_text = result[0]['generated_text']
        # 替换换行符为空格，避免多行问题
        return generated_text.replace('\n', ' ')
    except Exception as e:
        print(f"生成内容时出错（输入：{text}）：{e}")
        return ""  # 返回空字符串表示生成失败

# 读取CSV文件
input_csv = 'querys.csv'  # 输入文件路径

# 检测文件编码
with open(input_csv, 'rb') as f:
    result = chardet.detect(f.read())
    print(f"检测到文件编码: {result['encoding']}")

# 使用检测到的编码读取文件
df = pd.read_csv(input_csv, encoding=result['encoding'])

# 检查CSV文件是否包含所需的列
required_columns = ['original_query', 'obfuscated_query']
if not all(column in df.columns for column in required_columns):
    raise ValueError("CSV文件必须包含 'original_query', 'obfuscated_query' 列")

# 对每一列调用 GPT-Neo 生成内容
original_results = df['original_query'].apply(generate_content_gpt_neo)
obfuscated_results = df['obfuscated_query'].apply(generate_content_gpt_neo)

# 将生成结果添加到原始DataFrame中
# 添加新列
df['generated_original_gptneo'] = original_results.astype(str)
df['generated_obfuscated_gptneo'] = obfuscated_results.astype(str)

try:
    df.to_csv(input_csv, index=False, encoding='utf-8', mode='w')
    print("成功将生成结果添加到原文件：", input_csv)
except Exception as e:
    print(f"保存文件时出错：{e}")

