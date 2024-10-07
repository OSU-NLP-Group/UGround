import pandas as pd
import glob
import random
import json
import os
import io
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

def process_action_reprs(action_reprs):
    """
    Process action_reprs and convert it into a proper list format
    """
    if isinstance(action_reprs, np.ndarray):
        action_reprs = action_reprs.tolist()
    
    if isinstance(action_reprs, str):
        return action_reprs.split('\n')
    elif isinstance(action_reprs, list):
        if len(action_reprs) == 1 and isinstance(action_reprs[0], str):
            return action_reprs[0].split('\n')
        else:
            return action_reprs
    else:
        raise ValueError(f"Unexpected type for action_reprs: {type(action_reprs)}")

def process_split(split, input_dir, output_dir, samples_per_split):
    file_paths = glob.glob(f"{input_dir}/test_{split}*.parquet")
    dfs = []
    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        dfs.append(df)
    merged_df = pd.concat(dfs)
    merged_df = merged_df[(merged_df['screenshot'].apply(lambda x: x is not None and list(x.values())[0] is not None))]

    output_image_folder = f"{output_dir}/cross_{split}/images"
    os.makedirs(output_image_folder, exist_ok=True)

    selected_data = []
    unique_episodes = set()

    grouped = merged_df.groupby('annotation_id')

    all_annotation_ids = list(grouped.groups.keys())
    
    if samples_per_split:
        if samples_per_split > len(all_annotation_ids):
            samples_per_split = len(all_annotation_ids)
        sampled_annotation_ids = random.sample(all_annotation_ids, samples_per_split)
    else:
        sampled_annotation_ids = all_annotation_ids

    for annotation_id in sampled_annotation_ids:
        group = grouped.get_group(annotation_id)
        for _, row in group.iterrows():
            bbox_list = []
            for element_str in row['pos_candidates']:
                element_dict = json.loads(element_str)
                attributes_str = element_dict['attributes']
                attributes_dict = json.loads(attributes_str)
                bbox = tuple(map(float, attributes_dict['bounding_box_rect'].split(',')))
                bbox_list.append(bbox)

            screenshot_data = row['screenshot']
            image_data = list(screenshot_data.values())[0]

            image_stream = io.BytesIO(image_data)
            screenshot = Image.open(image_stream)
            image_filename = f"{row['annotation_id']}_{row['action_uid']}.png"
            screenshot.save(os.path.join(output_image_folder, image_filename))

            task = row["confirmed_task"]
            current_step = int(row["target_action_index"])
            action_reprs = process_action_reprs(row["action_reprs"])
            previous_actions = action_reprs[:current_step]

            operation_dict = json.loads(row['operation'])
            operation = operation_dict['op']
            value = operation_dict['value']

            total_steps = len(row["action_reprs"])
            website = row["website"]
            domain = row["domain"]
            subdomain = row["subdomain"]

            selected_data.append({
                "annotation_id": row['annotation_id'],
                "action_uid": row['action_uid'],
                "image": image_filename,
                "task": task,
                "website": website,
                "domain": domain,
                "subdomain": subdomain,
                "operation": operation,
                "value": value,
                "bbox": bbox_list,
                "previous_actions": previous_actions,
                "step": current_step,
                "total_steps": total_steps,
                "split": split
            })

            unique_episodes.add(row['annotation_id'])

    output_jsonl_file = f"{output_dir}/cross_{split}/sample.jsonl"
    os.makedirs(os.path.dirname(output_jsonl_file), exist_ok=True)

    with open(output_jsonl_file, 'w') as jsonl_file:
        for entry in selected_data:
            jsonl_file.write(json.dumps(entry) + "\n")

    print(f'Total tasks in cross_{split}: {len(unique_episodes)}')
    print(f'Total steps in cross_{split}: {len(selected_data)}')
    print(f'Done. Sample file saved at {output_jsonl_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Mind2Web data and generate JSONL output")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to Multimodal-Mind2Web dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for saving samples")
    parser.add_argument("--samples_per_split", type=int, default=None, help="Number of sample tasks per split (optional)")

    args = parser.parse_args()

    # Process each of the three splits
    for split in ['task', 'website', 'domain']:
        process_split(split, args.input_dir, args.output_dir, args.samples_per_split)