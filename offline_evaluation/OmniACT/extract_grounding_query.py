import json
import os
import argparse
from PIL import Image
import re

# Regular expression pattern for extracting actions
action_pattern = re.compile(
    r'pyautogui\.(click|rightClick|doubleClick|moveTo|dragTo|dragTo)\("([^"\\]*(?:\\.[^"\\]*)*)"'
    r'|pyautogui\.dragTo\("([^"]+)",'
)

def scale_image(image_path, long_edge=1344):
    """Scales the image to have the longer edge equal to the specified long_edge value."""
    image = Image.open(image_path)
    width, height = image.size
    if width > height:
        new_height = int((height / width) * long_edge)
        new_width = long_edge
    else:
        new_width = int((width / height) * long_edge)
        new_height = long_edge
    scale_factor = new_width / width
    return scale_factor

def extract_descriptions(gpt_output):
    """Extracts element descriptions from the GPT output using regex."""
    matches = action_pattern.findall(gpt_output)
    descriptions = [match[1] if match[0] else match[2] for match in matches if match[1] or match[2]]
    return descriptions

def process_file(base_path, result_file, seq_output_file, query_output_file):
    sequence_match = 0
    ideal_score = 0
    updated_records = []

    with open(result_file, 'r') as file:
        for line in file:
            record = json.loads(line.strip())
            task_path = os.path.join(base_path, record['task'])
            gpt_output = record['gpt_output']
        
            # Read gold script
            with open(task_path, 'r') as task_file:
                gold_script = task_file.read().strip().split('\n')[2:]
                gold_script = [line.lower() for line in gold_script]

            llm_script = [x for x in gpt_output.split('\n') if x.strip().startswith('pyautogui')]

            sample_weight = (len(gold_script) - 0.9)
            record['ideal_score'] = sample_weight
            ideal_score += sample_weight
            
            seq_match_flag = 1
            
            if len(llm_script) != len(gold_script):
                seq_match_flag = 0
            else:
                for i in range(len(gold_script)):
                    gold_line = gold_script[i].strip()
                    try:
                        gold_action = gold_line.split('pyautogui.')[1].split('(')[0]
                    except IndexError:
                        gold_action = None
                    
                    pred_line = llm_script[i].strip()
                    if not pred_line.startswith('pyautogui.'):
                        seq_match_flag = 0
                        break
                    pred_action = pred_line.split('pyautogui.')[1].split('(')[0]
                    if pred_action != gold_action:
                        seq_match_flag = 0
                        break
            
            sequence_match += seq_match_flag * sample_weight
            seq_score = seq_match_flag * sample_weight
            record['seq_score'] = seq_score

            updated_records.append(record)

    # Write seq score to the output file
    with open(seq_output_file, 'w') as seq_file:
        for record in updated_records:
            seq_file.write(json.dumps(record) + '\n')

    print(f"Seq score output file written to {seq_output_file}")
    print(f"Sequence match percentage: {sequence_match / ideal_score:.2%}")

    # Extract grounding queries
    with open(query_output_file, 'w') as query_file:
        for item in updated_records:
            image_path = os.path.join(base_path, item["image"])
            scale_factor = scale_image(image_path)
            if item["seq_score"] > 0:
                descriptions = extract_descriptions(item["gpt_output"])
                for description in descriptions:
                    jsonl_record = json.dumps({
                        'id': item['id'],
                        'image': item['image'],
                        'description': description,
                        'scale': scale_factor,
                    })
                    query_file.write(jsonl_record + '\n')

    print(f"Grounding query file written to {query_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sequence matching and extract grounding queries.")
    parser.add_argument('--plan_file', type=str, required=True, help="Path to the plan result JSONL file.")
    parser.add_argument('--base_path', type=str, required=True, help='Base directory for OmniACT dataset')
    parser.add_argument('--seq_output_file', type=str, required=True, help="Path to save the seq score output JSONL file.")
    parser.add_argument('--query_output_file', type=str, required=True, help="Path to save the grounding query JSONL file.")

    args = parser.parse_args()

    process_file(args.base_path, args.plan_file, args.seq_output_file, args.query_output_file)
