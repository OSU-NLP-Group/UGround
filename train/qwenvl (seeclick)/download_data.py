import boto3
import os

# 初始化S3客户端，不需要显式传递凭证
s3 = boto3.client('s3')



bucket_name = 'orby-osu-va'
key = 'boyugou/grounder_data/web_hy_qwen.parquet'

# 本地文件名
filename = os.path.basename(key)
local_path = os.path.join('.', filename)

# 下载对象到本地文件
s3.download_file(bucket_name, key, local_path)
print(f'Downloaded {key} to {local_path}')
