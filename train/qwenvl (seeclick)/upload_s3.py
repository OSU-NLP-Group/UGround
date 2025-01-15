import os
import ray
import boto3
from botocore.exceptions import NoCredentialsError

# 初始化Ray
ray.init()

# 获取S3客户端
def get_s3_client():
    try:
        s3_client = boto3.client('s3')
        return s3_client
    except NoCredentialsError:
        print("AWS credentials not found")
        return None

# 列出目录下的所有文件
def list_files(directory):
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn]

# 定义一个Ray远程函数来上传文件
@ray.remote
def upload_file(file_path, bucket_name, s3_key_prefix=""):
    s3_client = get_s3_client()
    if s3_client:
        # 生成正确的S3键
        relative_path = os.path.relpath(file_path, start=local_directory)
        s3_key = os.path.join(s3_key_prefix, relative_path)
        try:
            s3_client.upload_file(file_path, bucket_name, s3_key)
            return f"Uploaded: {file_path}"
        except Exception as e:
            return f"Failed to upload {file_path}: {e}"
    else:
        return "No S3 client available"

# 监控和上传文件
def upload_directory(directory, bucket_name, s3_key_prefix=""):
    uploaded_files_log = "uploaded_files.log"

    # 从日志中加载已经上传的文件
    if os.path.exists(uploaded_files_log):
        with open(uploaded_files_log, "r") as log_file:
            uploaded_files = set(log_file.read().splitlines())
    else:
        uploaded_files = set()

    file_paths = list_files(directory)
    upload_jobs = [upload_file.remote(file_path, bucket_name, s3_key_prefix) for file_path in file_paths if file_path not in uploaded_files]
    results = []

    with open(uploaded_files_log, "a") as log_file:
        while len(upload_jobs) > 0:
            done_id, upload_jobs = ray.wait(upload_jobs, num_returns=1)
            result = ray.get(done_id[0])
            results.append(result)
            if "Uploaded:" in result:
                uploaded_file = result.split("Uploaded: ")[1]
                log_file.write(uploaded_file + "\n")
            print(result)  # 或者记录到文件/数据库以便恢复

# 指定本地目录和S3桶的详细信息

'''
  export NCCL_MIN_NRINGS=4
  export NCCL_MAX_NRINGS=8
  export OUTPUTDIR=checkpoints/lora-llava-v1.5-vicuna-7b-16k-test-uiberttest-uiberttest-uibert
  export MACRO_NUM_EPOCH=1
  export MACRO_BATCH_SIZE=4
  export DATAPATHS3=small_data.parquet
'''
# OUTPUTDIR = str(os.environ['OUTPUTDIR'])
# DATAPATHS3 = str(os.environ['DATAPATHS3'])

local_directory = "merged_UGround_Qwen-web-hy"
s3_bucket_name = "orby-osu"
s3_prefix = f"boyugou/test_checkpoints/merged_UGround_Qwen-web-hy"

upload_directory(local_directory, s3_bucket_name, s3_prefix)


#   export NCCL_MAX_NRINGS=16