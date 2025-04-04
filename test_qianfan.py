import pandas as pd
import qianfan
import chardet

# 设置百度千帆的API密钥和Secret Key（新的配置方式）
config = qianfan.get_config()
config.AK = "8AdN1k6mkPgtBE2WMJe6aPVd"  # 替换为你的Access Key
config.SK = "RVPAK8srSdc3okIn6AqUUUGjmafG52y1"  # 替换为你的Secret Key

def generate_content(text):
    try:
        chat_comp = qianfan.ChatCompletion()
        resp = chat_comp.do(
            model="ERNIE-Bot",  # 使用ERNIE-Bot模型
            messages=[{"role": "user", "content": text}],
            temperature=0.7,
        )
        generated_text = resp['result']
        generated_text = generated_text.replace('\n', ' ')
        return generated_text
    except Exception as e:
        print(f"生成内容时出错（输入：{text}）：{e}")
        return ""

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



df['generated_query_qianfan'] = original_results.astype(str)



df.to_csv('test_qianfan_new.csv', index=False, encoding='utf-8', mode='w')

print("处理完成，结果已添加到原文件：", input_csv)
