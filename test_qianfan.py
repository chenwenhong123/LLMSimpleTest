import pandas as pd
import qianfan
import openai
import chardet

# 设置百度千帆的API密钥和Secret Key（新的配置方式）
config = qianfan.get_config()
config.AK = "yAKzhzXWpFQF8OuZrraEpEDF"  # 替换为你的Access Key
config.SK = "Dp88gdT9IBLKuTqjvLtiExn5Ms8qEBaV"  # 替换为你的Secret Key

# 定义生成内容的函数
def generate_content(text):
    """
    调用大模型生成内容
    """
    chat_comp = qianfan.ChatCompletion()
    resp = chat_comp.do(
        model="ERNIE-Bot",  # 使用文心一言模型
        messages=[{"role": "user", "content": text}],
        temperature=0.7,  # 控制生成内容的随机性
    )
    return resp['result']

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

# 对每一列调用大模型生成内容，并保存到新的DataFrame中
original_results = df['original_query'].apply(generate_content)
obfuscated_results = df['obfuscated_query'].apply(generate_content)
test_target_results = df['test_target'].apply(generate_content)

# 将结果保存到新的CSV文件中
original_results.to_csv('original_output_qianfan.csv', index=False, header=['generated_content'], encoding='utf-8')
obfuscated_results.to_csv('obfuscated_output_qianfan.csv', index=False, header=['generated_content'], encoding='utf-8')
test_target_results.to_csv('test_target_output_qianfan.csv', index=False, header=['generated_content'], encoding='utf-8')

print("处理完成，结果已保存到以下文件：")
print("- original_output_qianfan.csv")
print("- obfuscated_output_qianfan.csv")
print("- test_target_output_qianfan.csv")