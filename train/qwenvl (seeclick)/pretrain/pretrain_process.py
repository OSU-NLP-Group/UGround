import json
import task_prompts
from tqdm import tqdm
import os
import random
import argparse
from PIL import Image
import re
import random
import string
from concurrent.futures import ThreadPoolExecutor

# Generate a random string of 26 letters
def resize_coordinates(old_coords, old_size, new_size=None):
    """
    根据图片 resize 的比例调整坐标。
    """
    point = [old_coords[0] / old_size[0], old_coords[1] / old_size[1]]
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


def process_item(item):
    formatted_data = []
    with Image.open(item["image"]) as img:
        original_size = img.size

    img_filename = item["image"]
    img_path = img_filename
    conversations = []
    data_conversations = item["conversations"]

    prompt = random.choice(task_prompts.web_loca_all_point_prompt)
    prompt += ' '
    # print(data_conversations)

    for j, conversation in enumerate(data_conversations):
        if j == 0:
            # print(conversation)
            # print(conversation["value"])
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_user["value"] += extract_corresponding_content(conversation["value"])
            conversations.append(conv_user)
        elif conversation["from"] == 'human':
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += extract_corresponding_content(conversation["value"])
            conversations.append(conv_user)
        else:
            value = conversation["value"]
            old_coords = tuple(map(int, value.strip("()").split(",")))
            new_coords = resize_coordinates(old_coords, original_size)
            conv_ai = {"from": "assistant", "value": new_coords}
            conversations.append(conv_ai)

    random_string = ''.join(random.choices(string.ascii_lowercase, k=30))
    formatted_data = {"id": f"{random_string}", "conversations": conversations}

    return formatted_data

def main():
    json_file_path = '/fs/ess/PAS1576/boyu_gou/data/web_hy_mobile_and_web/reversed_web_hy.json'
    output_file_path = '/fs/ess/PAS1576/boyu_gou/projects/UGround/paper_ablation/SeeClick/data/sft_train.json'

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    formatted_datapoints = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(process_item, data), total=len(data), desc="Processing"))

    formatted_datapoints = [result for result in results if result]

    print("Num of sft: " + str(len(formatted_datapoints)))

    with open(output_file_path, 'w') as f:
        json.dump(formatted_datapoints, f)

if __name__ == "__main__":
    main()
