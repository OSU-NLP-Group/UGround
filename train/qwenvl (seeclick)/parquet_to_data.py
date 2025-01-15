import json
import pyarrow.parquet as pq
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import string
from PIL import Image
import io
import os
import random
from datetime import datetime


# locate all elements in a webpage (point)
web_loca_all_point_prompt = [
    "In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions (with point).",
    "Based on the screenshot of the page, I give a text description and you give its corresponding location (with point).",
    "In the image above, I will give a series of descriptions of the elements to be clicked. Please predict where you want to click (with point).",
    "I will give textual descriptions of certain elements in the screenshot. Please predict the location of the corresponding element (with point).",
    "Please identify the coordinates of the webpage elements I describe based on the provided screenshot (with point).",
    "Given a screenshot, I will describe specific elements; your task is to predict their locations (with point).",
    "Using the image of this webpage, can you determine the coordinates of the elements I describe (with point)?",
    "In this webpage capture, I will describe certain elements. Please locate them for me (with point).",
    "I'll provide textual descriptions of elements in this webpage screenshot. Can you find their coordinates (with point)?",
    "From the given webpage screenshot, I need you to identify the locations of described elements (with point).",
    "Based on this screenshot, I'll describe some elements. Please pinpoint their exact locations (with point).",
    "For the elements I describe in this page capture, can you predict their positions (with point)?",
    "I will describe elements from a webpage screenshot; your role is to locate them (with point).",
    "Using the attached screenshot of a webpage, please find the coordinates of described elements (with point).",
    "From the image of this webpage, I will describe elements for you to locate (with point).",
    "I'll give descriptions of certain webpage elements; please identify where they are in this screenshot (with point).",
    "On this webpage screenshot, I will point out elements; please predict their exact coordinates (with point).",
    "In this web page image, please locate the elements as I describe them (with point).",
    "Given this screenshot of a webpage, I'll describe some elements; locate them for me (with point).",
    "Please use the provided webpage screenshot to locate the elements I describe (with point).",
    "In the provided web page image, I'll describe specific elements. Identify their locations, please (with point).",
    "With this screenshot of a webpage, can you locate the elements I describe (with point)?",
    "I will describe features on this webpage screenshot; please predict their positions (with point).",
    "Using the screenshot of this webpage, identify the coordinates of elements I describe (with point).",
    "On this webpage capture, I'll point out specific elements for you to locate (with point).",
    "Please determine the location of elements I describe in this webpage screenshot (with point).",
    "I'll describe certain elements on this webpage image; your task is to find their locations (with point).",
    "Using this webpage screenshot, I'll describe some elements. Please locate them (with point).",
    "Based on my descriptions, find the locations of elements in this webpage screenshot (with point).",
    "In this web page capture, please predict the positions of elements I describe (with point).",
    "I'll give textual clues about elements in this webpage screenshot; identify their coordinates (with point).",
    "Using the provided screenshot, I'll describe webpage elements for you to locate (with point).",
    "From this webpage image, I will describe specific elements. Please predict their exact locations (with point)."
]


# Generate a random string of 26 letters
def resize_coordinates(old_coords):
    """
    根据图片 resize 的比例调整坐标。
    """
    point = [old_coords[0] / 448, old_coords[1] / 448]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str




def extract_corresponding_content(text):
    """
    Extract the content between the first " after 'corresponding to' and the last " in the string.

    Args:
        text (str): The input string containing the target content.

    Returns:
        str: The extracted content or None if no match is found.
    """
    # Find the part after 'corresponding to'
    pattern = r'corresponding to\s+"'
    match = re.search(pattern, text)

    if not match:
        raise Exception

    # Extract substring starting after 'corresponding to "'
    start_idx = match.end() - 1  # Include the first "
    substring = text[start_idx:]

    # Find the last " in the substring
    last_quote_idx = substring.rfind('"')

    if last_quote_idx == -1:
        raise Exception

    # Extract content between the first and last quote
    return substring[1:last_quote_idx]



def extract_img_path(text):
    """
    Extract the file path enclosed in <img> and </img> tags from the given text.

    Args:
    text (str): The input string containing <img> and </img> tags.

    Returns:
    list: A list of file paths found between <img> and </img> tags.
    """
    pattern = r"<img>(.*?)</img>"
    img_paths = re.findall(pattern, text)
    return img_paths

import random
def process_item(item):


    image_id=item['id']
    image_bytes=item['image']

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    data_conversations=json.loads(item['conversations'])

    os.makedirs("images", exist_ok=True)

    # Save image to local directory with timestamp
    timestamp = datetime.now().strftime("%M%S")
    image_filename = f"{image_id}_{timestamp}.png"
    image_save_abs_path = os.path.abspath(os.path.join("images", image_filename))
    image.save(image_save_abs_path)

    data_conversations[0]["value"]=data_conversations[0]["value"].replace("PPPLLLAAACCCHOLLDER",image_save_abs_path)


    record = {
        "id": f"{image_filename[:-4]}",
        "conversations": json.dumps(data_conversations)  # 序列化为JSON字符串
    }
    return record


# 输入和输出 Parquet 文件路径
input_parquet_path = 'web_hy_qwen.parquet'
output_file_path = 'data/sft_train.json'

# 分块读取参数
chunk_size = 10000  # 暂时设置为 12作为小规模试验
cpu_count = 10

# 初始化写入器


# reader = pq.ParquetFile(input_parquet_path)  # 初始化 PyArrow 文件读取对象
# num_batches = (reader.metadata.num_rows + chunk_size - 1) // chunk_size  # 计算批次数
# print(num_batches)
# # 遍历每个分块
#
# records = []








records = []
table = pq.read_table(input_parquet_path)
num_rows = table.num_rows
df = table.to_pandas()

for start_idx in range(0, num_rows, chunk_size):
    end_idx = min(start_idx + chunk_size, num_rows)
    chunk_data = df.iloc[start_idx:end_idx]


    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        future_to_record = {executor.submit(process_item, row): row for _, row in chunk_data.iterrows()}
        for future in tqdm(as_completed(future_to_record), total=len(future_to_record),
                           desc=f"Processing Rows {start_idx + 1}-{end_idx}/{num_rows}"):
            records.append(future.result())





with open(output_file_path, 'w') as f:
    json.dump(records, f)

print("完成处理并保存到新 Parquet 文件中。")
