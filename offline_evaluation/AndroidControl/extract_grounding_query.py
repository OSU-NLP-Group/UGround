import json
import os
import argparse
from PIL import Image

def is_action_match(plan_action, sample_action):
    plan_action_type = plan_action.get("action_type")
    sample_action_type = sample_action.get("action_type")

    condition_1 = plan_action_type in ["click", "long_press", "type_text"] and sample_action_type == plan_action_type

    condition_2 = plan_action_type == "click" and sample_action_type in ["navigate_back", "open_app"]

    condition_3 = plan_action_type in ["navigate_back", "open_app"] and sample_action_type == "click"

    return condition_1 or condition_2 or condition_3
    
def process_files(sample_file, plan_file, output_file, screenshot_dir):
    with open(sample_file, 'r') as f:
        sample_data = json.load(f)

    sample_dict = {(entry['episode_id'], entry['step']): entry for entry in sample_data}

    with open(output_file, 'w') as out_f:
        with open(plan_file, 'r') as plan_f:
            for line in plan_f:
                plan_entry = json.loads(line.strip())
                episode_id = plan_entry['episode_id']
                step = plan_entry['step']
                
                sample_entry = sample_dict.get((episode_id, step))

                if not plan_entry.get("action"):
                    continue
                
                if sample_entry:
                    if is_action_match(plan_entry["action"], sample_entry["action"]):
                        image_path = os.path.join(screenshot_dir, sample_entry["screenshot"])

                        new_entry = {
                            "episode_id": episode_id,
                            "step": step,
                            "image": sample_entry["screenshot"],
                            "description": plan_entry["action"].get("element", ""),
                        }
                        
                        json.dump(new_entry, out_f)
                        out_f.write('\n')

    print(f"Processed data has been written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sample and plan files and generate output in JSONL format")
    parser.add_argument("--sample_file", type=str, required=True, help="Path to sample JSON file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to plan JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--screenshot_dir", type=str, required=True, help="Directory where screenshot images are stored")

    args = parser.parse_args()

    process_files(args.sample_file, args.plan_file, args.output_file, args.screenshot_dir)
