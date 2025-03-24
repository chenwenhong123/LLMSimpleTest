import pandas as pd
from transformers import pipeline, BloomTokenizerFast
import chardet
import torch

# 加载 BLOOM 文本生成模型和 tokenizer，并指定设备
device = "cuda" if torch.cuda.is_available() else "cpu"
bloom_generator = pipeline("text-generation", model="bigscience/bloom-560m", device=device)
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")

# 定义生成内容的函数
def generate_content_bloom(text):
    input_length = len(tokenizer.encode(text))
    max_length = input_length + 500  # 增加max_length的值
    result = bloom_generator(
        text,
        max_length=max_length,
        num_return_sequences=1,
        truncation=True,
    )
    generated_text = result[0]['generated_text']
    generated_text = generated_text.replace('\n', ' ')
    return generated_text

# 读取CSV文件
input_csv = 'texts.csv'

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

# 对每一列调用 BLOOM 生成内容
original_results = df['original_query'].apply(generate_content_bloom)
obfuscated_results = df['obfuscated_query'].apply(generate_content_bloom)

# 将结果添加到DataFrame，并保存到原CSV文件
df['generated_original_bloom'] = original_results
df['generated_obfuscated_bloom'] = obfuscated_results

# 保存回原CSV文件，使用UTF-8编码
df.to_csv(input_csv, index=False, encoding='utf-8', mode='w')

print("处理完成，结果已添加到原始文件：", input_csv)
