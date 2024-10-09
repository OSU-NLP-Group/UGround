import json
import random
import re
import os
import argparse

def replace_screen_pattern(value):
    image_pattern = re.compile(r'screen_(\d+)\.png')
    box_pattern = re.compile(r'screen_(\d+)\.json')
    
    value['image'] = image_pattern.sub(r'screen\1.png', value['image'])
    value['box'] = box_pattern.sub(r'screen\1_boxes.json', value['box'])
    
    return value

# Main script function
def main(args):
    test_file = args.test_file
    base_path = args.base_path
    output_file = args.output_file
    sample_num = args.sample_num

    with open(test_file, 'r') as f:
        data = json.load(f)

    if sample_num is not None and 0 < sample_num < len(data):
        selected_samples = random.sample(list(data.items()), sample_num)
    else:
        selected_samples = list(data.items())

    # Replace wrong box and image path and check
    for key, value in selected_samples:
        if 'web' in value['task']:
            value = replace_screen_pattern(value)

    for key, value in selected_samples:
        image_path = os.path.join(base_path, value['image'])
        box_path = os.path.join(base_path, value['box'])
        if not os.path.exists(image_path):
            print(f"Warning: Image path does not exist: {value['image']}")
        if not os.path.exists(box_path):
            print(f"Warning: Box path does not exist: {value['box']}")

    selected_samples_dict = {str(key): value for key, value in selected_samples}

    with open(output_file, 'w') as f:
        json.dump(selected_samples_dict, f, indent=4)

    print('JSON file saved successfully at', output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample and process OmniACT test data")
    parser.add_argument('--base_path', type=str, required=True, help="Base path to the OmniACT dataset")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the OmniACT origin test JSON file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSON file")
    parser.add_argument('--sample_num', type=int, default=None, help="Number of samples to select (optional)")

    args = parser.parse_args()
    main(args)
