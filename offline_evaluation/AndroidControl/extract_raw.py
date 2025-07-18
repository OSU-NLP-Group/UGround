import base64
import io
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
from android_env_utils import representation_utils
from android_env_utils.android_env.proto.a11y import android_accessibility_forest_pb2

# import pandas as pd
from PIL import Image
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct file paths relative to the script directory
# Load train episode IDs from split.json
split_file = os.path.join(script_dir, "android_control/splits.json")
with open(split_file, "r") as f:
    split_data = json.load(f)
train_episode_ids = set(split_data["train"])

# Define the feature description for parsing TFRecord files
feature_description = {
    "episode_id": tf.io.FixedLenFeature([], tf.int64),
    "goal": tf.io.FixedLenFeature([], tf.string),
    "screenshots": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "accessibility_trees": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "screenshot_widths": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "screenshot_heights": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "actions": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "step_instructions": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
}


# Function to parse a single example from the TFRecord file
def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


# Convert UI elements to dictionaries for easier handling
def convert_ui_elements_to_dicts(ui_elements):
    filtered_list = []
    """ The filtering implementations in Android Control and Android World are a bit different:
    Android World:
    https://github.com/google-research/android_world/blob/4d941153396c5daf4adc013f2f5cd265858bdffa/android_world/agents/m3a_utils.py#L448

    Android Control:
    https://github.com/google-research/google-research/issues/2120#issuecomment-2332316904

    """

    # Here I use AndroidWorld's for now since it makes more senses:

    for element in ui_elements:
        element_dict = element.to_dict()
        if (
            (element_dict["content_description"] is not None or element_dict["text"] is not None)
            and element_dict["is_visible"]
            and (element_dict["bbox_pixels"]["height"] > 0)
            and (element_dict["bbox_pixels"]["width"] > 0)
            and (element_dict["bbox_pixels"]["x_min"] < 1080)
            and (element_dict["bbox_pixels"]["y_min"] < 2400)
            and (element_dict["bbox_pixels"]["x_max"] > 0)
            and (element_dict["bbox_pixels"]["y_max"] > 0)
        ):
            filtered_list.append(element_dict)

    return filtered_list


# Function to process a single TFRecord file
def process_tfrecord_file(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type="GZIP")
    parsed_dataset = raw_dataset.map(_parse_function)
    file_data = []

    for parsed_record in parsed_dataset:
        record = {key: parsed_record[key].numpy() for key in feature_description.keys()}
        episode_id = int(record["episode_id"])

        # Skip records not in train_episode_ids
        if episode_id not in train_episode_ids:
            continue

        # Convert bytes to appropriate data types
        record["goal"] = record["goal"].decode("utf-8")
        record["actions"] = [json.loads(action) for action in record["actions"]]
        record["step_instructions"] = [
            step_instruction.decode("utf-8") for step_instruction in record["step_instructions"]
        ]

        # Handle accessibility_trees using base64 encoding
        if "accessibility_trees" in record:
            parsed_trees = []
            for tree_data in record["accessibility_trees"]:
                forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
                forest.ParseFromString(tree_data)
                ui_elements = representation_utils.forest_to_ui_elements(forest)
                parsed_trees.append(convert_ui_elements_to_dicts(ui_elements))
            record["accessibility_trees"] = parsed_trees

        # Save screenshots as images
        screenshots_dir = os.path.join(output_dir, str(episode_id))
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        for i, screenshot in enumerate(record["screenshots"]):
            image = Image.open(io.BytesIO(screenshot))
            image_path = os.path.join(screenshots_dir, f"screenshot_{i}.png")
            image.save(image_path)

        # Add file paths to the record
        record["screenshots"] = [
            os.path.join(str(episode_id), f"screenshot_{i}.png").replace("\\", "/")
            for i in range(len(record["screenshot_widths"]))
        ]

        # Convert NumPy data types to Python native types
        for key, value in record.items():
            if isinstance(value, bytes):
                record[key] = base64.b64encode(value).decode("utf-8")
            elif isinstance(value, np.ndarray):
                record[key] = value.tolist()
            elif isinstance(value, np.int64):
                record[key] = int(value)

        file_data.append(record)

    return file_data


# Directory containing the TFRecord files
data_dir = os.path.join(script_dir, "android_control")
output_dir = os.path.join(script_dir, "android_control_screenshots")
# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Get the list of TFRecord files
tfrecord_files = [
    os.path.join(data_dir, f) for f in os.listdir(data_dir) if not f.endswith(".json")
]
tfrecord_files_remote = tf.io.gfile.glob("gs://gresearch/android_control/android_control*")
# Create a list to store parsed data
data = []
# Use ThreadPoolExecutor to process files in parallel
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {
        executor.submit(process_tfrecord_file, tfrecord_file): tfrecord_file
        for tfrecord_file in tfrecord_files
    }
    # TODO: If you want to skip downloading the tf record files, you can use 'tfrecord_files_remote' instead of 'tfrecord_files'
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing TFRecord files"):
        try:
            file_data = future.result()
            data.extend(file_data)
        except Exception:
            traceback.print_exc()
            # print(f"Error processing file {futures[future]}: {e}")

# print(f"Total records processed: {len(data)}")

# Save data to a JSON file
# TODO: If you want to directly dump all the files
with open("sample_raw.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
# TODO: If you want to print them out. Notice that the dataset is super large!
for i in data:
    print(json.dumps(i))

# Uncomment the following lines if you want to save the data to a CSV file
# import pandas as pd
# df = pd.DataFrame(data)
# df.to_csv('sample_raw.csv', index=False)
