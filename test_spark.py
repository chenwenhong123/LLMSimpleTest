import pandas as pd
import chardet
import requests
import json
import time
import hashlib
import base64
import hmac
from urllib.parse import urlencode

# 讯飞星火的API配置
APPID = "f75fe76d"
APIKey = "31dcda393e0948f90ebc9257c0bac70e"
APISecret = "YzJhOGY0OTE4NTUyMjdhN2E4YzM4NTBk"
API_URL = "wss://spark-api.xf-yun.com/v4.0/chat"

# 生成请求头
def get_header():
    # 生成时间戳
    timestamp = str(int(time.time()))
    # 生成签名字符串
    signature_origin = f"host: spark-api.xf-yun.com\ndate: {timestamp}\nGET /v4/chat/completions HTTP/1.1"
    # 使用HMAC-SHA256算法生成签名
    signature_sha = hmac.new(APISecret.encode('utf-8'), signature_origin.encode('utf-8'), hashlib.sha256).digest()
    signature_sha_base64 = base64.b64encode(signature_sha).decode('utf-8')
    # 生成Authorization
    authorization_origin = f'api_key="{APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
    # 生成请求头
    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json",
        "Host": "spark-api.xf-yun.com",
        "Date": timestamp
    }
    return headers

# 定义生成内容的函数
def generate_content(text):
    """
    调用讯飞星火大模型生成内容
    """
    try:
        headers = get_header()
        data = {
            "header": {
                "app_id": APPID,
                "uid": "12345"  # 用户ID，可以自定义
            },
            "parameter": {
                "chat": {
                    "domain": "general",
                    "temperature": 0.7,  # 控制生成内容的随机性
                    "max_tokens": 1024  # 最大生成长度
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {"role": "user", "content": text}
                    ]
                }
            }
        }
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            return result['payload']['choices']['text'][0]['content']
        else:
            print(f"API请求失败: {response.status_code}, {response.text}")
            return None  # 返回 None 表示失败
    except Exception as e:
        print(f"生成内容时发生异常: {e}")
        return None  # 返回 None 表示失败

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
original_results.to_csv('original_output_spark.csv', index=False, header=['generated_content'], encoding='utf-8')
obfuscated_results.to_csv('obfuscated_output_spark.csv', index=False, header=['generated_content'], encoding='utf-8')
test_target_results.to_csv('test_target_output_spark.csv', index=False, header=['generated_content'], encoding='utf-8')

print("处理完成，结果已保存到以下文件：")
print("- original_output_spark.csv")
print("- obfuscated_output_spark.csv")
print("- test_target_output_spark.csv")