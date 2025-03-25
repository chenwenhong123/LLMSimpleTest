import pandas as pd

df1 = pd.read_csv('test_gptneo_new_1.3B.csv')
df2 = pd.read_csv('test_qianfan_new.csv')
df3 = pd.read_csv('test_bloom_new.csv')
df4 = pd.read_csv('test_glm_new.csv')
df5 = pd.read_csv('test_glmflash_new.csv')

# 合并两个DataFrame
merged_df = pd.DataFrame({
    'id': df1['id'],
    'query': df1['query'],
    'generated_gptneo': df1['generated'],
    'generated_qianfan': df2['generated_query_qianfan'],
    'generated_bloom': df3['generated_content'],
    'generated_glm': df4['generated_glm4'],
    'generated_glmflash': df5['generated_glm4flash']
})

# 保存合并后的DataFrame到新的CSV文件
merged_df.to_csv('merged_responses.csv', index=False, encoding='utf-8-sig')
print("合并完成")