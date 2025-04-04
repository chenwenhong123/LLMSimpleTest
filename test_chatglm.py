import pandas as pd
from zhipuai import ZhipuAI
import chardet

# 初始化智谱AI客户端
client = ZhipuAI(api_key="ff39e47dc162478d8bd81fc5db870862.JHinw3TOCqIHaq5v")  # 替换为你的智谱AI API Key

def generate_content(text):
    try:
        response = client.chat.completions.create(
            model="glm-4",  # 使用GLM-4 Flash模型
            messages=[
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        generated_text = response.choices[0].message.content.strip()
        generated_text = generated_text.replace('\n', ' ')
        return generated_text
    except Exception as e:
        print(f"生成内容时出错（输入：{text}）：{e}")
        return None  # 返回None表示生成失败

# 读取CSV文件
input_csv = 'querys.csv'  # 输入文件路径

# 检测文件编码
with open(input_csv, 'rb') as f:
    result = chardet.detect(f.read())
    print(f"检测到文件编码: {result['encoding']}")

# 使用检测到的编码读取文件
df = pd.read_csv(input_csv, encoding=result['encoding'])

# 检查CSV文件是否包含所需的列
required_columns = ['query']
if not all(column in df.columns for column in required_columns):
    raise ValueError("CSV文件必须包含 'query' 列")

# 对每一列调用大模型生成内容
original_results = df['query'].apply(generate_content)

df['generated_glm4'] = original_results.astype(str)

df.to_csv('test_glm_new.csv', index=False, encoding='utf-8', mode='w')

print("处理完成")