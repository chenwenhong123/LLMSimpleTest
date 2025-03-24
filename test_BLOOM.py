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
    """
    使用 BLOOM 生成内容
    """
    # 动态计算 max_length
    input_length = len(tokenizer.encode(text))
    max_length = input_length + 50  # 输入长度 + 50 个 token
    result = bloom_generator(text, max_length=max_length, num_return_sequences=1, truncation=True)
    return result[0]['generated_text']

# 读取CSV文件
input_csv = 'texts.csv'  # 输入文件路径

# 检测文件编码
with open(input_csv, 'rb') as f:
    result = chardet.detect(f.read())
    print(f"检测到文件编码: {result['encoding']}")

# 使用检测到的编码读取文件
df = pd.read_csv(input_csv, encoding=result['encoding'])

# 检查CSV文件是否包含所需的列
required_columns = ['original_query', 'obfuscated_query', 'test_target']
if not all(column in df.columns for column in required_columns):
    raise ValueError("CSV文件必须包含 'original_query', 'obfuscated_query', 'test_target' 列")

# 对每一列调用 BLOOM 生成内容
original_results = df['original_query'].apply(generate_content_bloom)
obfuscated_results = df['obfuscated_query'].apply(generate_content_bloom)
test_target_results = df['test_target'].apply(generate_content_bloom)

# 将结果保存到新的CSV文件中（使用 UTF-8 编码）
original_results.to_csv('original_output_bloom.csv', index=False, header=['generated_content'], encoding='utf-8')
obfuscated_results.to_csv('obfuscated_output_bloom.csv', index=False, header=['generated_content'], encoding='utf-8')
test_target_results.to_csv('test_target_output_bloom.csv', index=False, header=['generated_content'], encoding='utf-8')

print("处理完成，结果已保存到以下文件：")
print("- original_output_bloom.csv")
print("- obfuscated_output_bloom.csv")
print("- test_target_output_bloom.csv")