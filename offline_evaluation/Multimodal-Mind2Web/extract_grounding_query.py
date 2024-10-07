import json
import os
from tqdm import tqdm
from PIL import Image
import argparse

def scale_image(image_path):
    """Scale the image to a fixed width of 1344 pixels, maintaining aspect ratio."""
    image = Image.open(image_path)
    original_width, original_height = image.size
    new_width = 1344
    scale_factor = 1344 / original_width
    new_height = original_height * scale_factor

    return scale_factor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Mind2Web data and generate JSONL output")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input plan JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output query JSONL file")
    parser.add_argument("--blocks", type=str, required=True, help="Base directory for block images")
    
    args = parser.parse_args()

    input_file_path = args.input_file
    output_file_path = args.output_file
    block_image_dir = args.blocks

    # Read input file
    with open(input_file_path, 'r') as infile:
        data = [json.loads(line) for line in infile if line.strip()]

    # Write to output file
    with open(output_file_path, 'w') as outfile:
        for item in data:
            if str(item["ans_block"]) in item["target_blocks"].keys():
                # Skip if description is missing or 'None'
                if not item["description"] or item["description"] == "None":
                    continue
                block_name = f"{str(item['ans_block'])}.png"
                image_path = os.path.join(block_image_dir, item['blocks_path'], block_name)
                box = item["target_blocks"].get(str(item["ans_block"]))
                scale_factor = scale_image(image_path)
                jsonl_record = json.dumps({
                    'id': item['blocks_path'],
                    'image': os.path.join(item['blocks_path'], block_name),
                    'bbox': box,
                    'description': item["description"],
                    'scale': scale_factor,
                })
                outfile.write(jsonl_record + '\n')

    print(f'Done. JSONL file has been saved to {output_file_path}')
