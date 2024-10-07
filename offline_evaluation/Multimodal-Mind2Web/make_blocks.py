import os
import json
import base64
import pandas as pd
from PIL import Image
from tqdm import tqdm
import math
import random
from multiprocessing import Pool, cpu_count
import argparse

def generate_screenshots(screenshot, bbox_list, action_uid, output_size, padding):
    """Split the entire screenshot into blocks of the specified output size."""
    full_width, full_height = screenshot.size

    # If bbox_list is empty, return the first block of the screenshot
    if not bbox_list:
        first_block = screenshot.crop((0, 0, min(full_width, output_size[0]), min(full_height, output_size[1])))
        return [first_block], {"0": []}

    if full_width <= 1280:
        output_size[0] = full_width
    else:
        output_size[0] = 1280

    num_blocks_y = (full_height - padding) // (output_size[1] - padding) + 1

    screenshots = []
    all_bbox_list = []

    for i in range(num_blocks_y):
        start_x = 0
        end_x = output_size[0]
        start_y = max(0, i * (output_size[1] - padding))
        end_y = min(start_y + output_size[1], full_height)

        block_screenshot = screenshot.crop((start_x, start_y, end_x, end_y))
        screenshots.append((start_y, block_screenshot))

        block_bbox_list = []
        for bbox in bbox_list:
            x, y, width, height = bbox
            block_bbox = (x - start_x, y - start_y, width, height)
            block_bbox_list.append(block_bbox)
        
        all_bbox_list.append(block_bbox_list)

    target_blocks = {}
    for i, block_bbox_list in enumerate(all_bbox_list):
        tmp_bboxes = []
        for bbox in block_bbox_list:
            if 0 <= bbox[0] < output_size[0] and 0 <= bbox[0] + bbox[2] <= output_size[0] and \
               0 <= bbox[1] < output_size[1] and 0 <= bbox[1] + bbox[3] <= output_size[1]:
                tmp_bboxes.append(bbox)
        if tmp_bboxes:
            target_blocks[i] = tmp_bboxes

    # If no block contains the target element, create an additional screenshot
    if not target_blocks:
        min_x = min(bbox[0] for bbox in bbox_list)
        min_y = min(bbox[1] for bbox in bbox_list)
        max_x = max(bbox[0] + bbox[2] for bbox in bbox_list)
        max_y = max(bbox[1] + bbox[3] for bbox in bbox_list)

        element_width = max_x - min_x
        element_height = max_y - min_y

        if element_height < output_size[1]:
            max_start_y = min(full_height - output_size[1], min_y)
            start_y = random.randint(max(0, int(min_y - output_size[1] + element_height)), int(max_start_y))
            height = output_size[1]
        else:
            start_y = min_y
            height = min(element_height, output_size[1])

        if max_x > output_size[0]:
            start_x = max_x + random.randint(1, 30) - output_size[0]
        else:
            start_x = 0

        start_y = min(max(0, start_y), full_height - height)

        additional_screenshot = screenshot.crop((start_x, start_y, start_x + element_width, start_y + height))
        screenshots.append((start_y, additional_screenshot))

        additional_block_bbox_list = []
        for bbox in bbox_list:
            x, y, w, h = bbox
            new_bbox = (x - start_x, y - start_y, w, h)
            additional_block_bbox_list.append(new_bbox)
        all_bbox_list.append(additional_block_bbox_list)

    # Sort screenshots by start_y
    screenshots.sort(key=lambda x: x[0])
    all_bbox_list.sort(key=lambda x: x[0][1], reverse=True)

    # Create new target_blocks to ensure correct indices and coordinates
    new_target_blocks = {}
    for i, ((start_y, _), block_bbox_list) in enumerate(zip(screenshots, all_bbox_list)):
        tmp_bboxes = []
        for bbox in block_bbox_list:
            if 0 <= bbox[0] < output_size[0] and 0 <= bbox[0] + bbox[2] <= output_size[0] and \
               0 <= bbox[1] < output_size[1] and 0 <= bbox[1] + bbox[3] <= output_size[1]:
                tmp_bboxes.append(bbox)
        if tmp_bboxes:
            new_target_blocks[i] = tmp_bboxes

    screenshots = [screenshot for _, screenshot in screenshots]

    assert len(new_target_blocks) > 0, f"Failed to create a block containing the target element for action_uid: {action_uid}, {bbox_list}"

    return screenshots, new_target_blocks

def process_single_item(args):
    item, split, image_folder, output_folder, output_size, padding = args
    annotation_id = item['annotation_id']
    action_uid = item['action_uid']
    image_file = item['image']
    bbox_list = item['bbox']

    image_path = os.path.join(image_folder, image_file)
    target_blocks = {}
    try:
        screenshot = Image.open(image_path)
        screenshots, target_blocks = generate_screenshots(screenshot, bbox_list, action_uid, output_size, padding)
        if screenshots:
            block_dir = os.path.join(output_folder, f"{annotation_id}_{action_uid}")
            os.makedirs(block_dir, exist_ok=True)
            for i, block_screenshot in enumerate(screenshots):
                block_screenshot.save(os.path.join(block_dir, f"{i}.png"))
        else:
            return None

    except OSError as e:
        print(f"Error processing {annotation_id} with action {action_uid} for image {image_path}: {e}")
        return None

    del item["image"]
    item["blocks_path"] = f"{annotation_id}_{action_uid}"
    item["target_blocks"] = target_blocks

    return item

def main():
    parser = argparse.ArgumentParser(description="Process Mind2Web data with specified split.")
    parser.add_argument("--input_file", required=True, help="Path to the sample JSONL file")
    parser.add_argument("--output_file", required=True, help="Path to save the sample (blocks) JSONL file")
    parser.add_argument("--image_folder", required=True, help="Path to the folder containing original screenshots")
    parser.add_argument("--output_folder", required=True, help="Folder to save the processed image blocks")
    parser.add_argument("--output_size", nargs=2, type=int, default=[1280, 1000], help="Output size for the blocks, e.g., 1280 720")
    parser.add_argument("--padding", type=int, default=200, help="Padding between blocks")
    
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    image_folder = args.image_folder
    output_folder = args.output_folder
    output_size = args.output_size
    padding = args.padding

    try:
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    num_cores = cpu_count() - 1  # Use all but one core
    pool_args = [(item, 'task', image_folder, output_folder, output_size, padding) for item in data]
    
    with Pool(num_cores) as pool:
        results = [result for result in tqdm(pool.imap(process_single_item, pool_args), total=len(data)) if result is not None]

    try:
        with open(output_file, "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
            f.flush()
    except Exception as e:
        print(f"Error writing output file: {e}")
        return

    print('File saved at', output_file)

if __name__ == "__main__":
    main()
