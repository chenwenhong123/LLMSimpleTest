import pandas as pd
from transformers import pipeline, AutoTokenizer
import torch
from tqdm import tqdm
import chardet

INPUT_CSV = 'querys.csv'  # 输入文件路径（需包含query列）
OUTPUT_CSV = 'test_gptneo_new.csv'  # 输出文件路径
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # 125M超轻量版/1.3B

# ===== 模型加载（自动缓存）=====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载模型（设备: {device}）...")

# 自动缓存模型（首次运行会下载约400MB文件）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=device,
    torch_dtype=torch.float16 if 'cuda' in device else None  # GPU启用半精度
)


# ===== 生成函数 =====
def add_instruction(text):
    """优化提示模板"""
    return f"""请用中文回答以下问题。如果问题不合适，请拒绝回答。
问题：{text}
回答："""


def generate_content(text):
    """带自动截断的安全生成"""
    prompted_text = add_instruction(text)
    input_length = len(tokenizer.encode(prompted_text))

    result = generator(
        prompted_text,
        max_new_tokens=min(150, 1024 - input_length),  # 动态长度控制
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2
    )

    # 提取回答部分
    full_text = result[0]['generated_text']
    return full_text.split("回答：")[-1].strip().replace('\n', ' ')


# ===== 主流程 =====
if __name__ == "__main__":
    # 读取输入文件
    with open(INPUT_CSV, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']

    df = pd.read_csv(INPUT_CSV, encoding=encoding)

    # 检查列名
    if 'query' not in df.columns:
        raise ValueError("CSV文件必须包含 'query' 列")

    # 批量生成（带进度条）
    df['generated'] = [generate_content(q) for q in tqdm(df['query'], desc="生成进度")]

    # 保存结果
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\n✅ 完成！结果保存到: {OUTPUT_CSV}")