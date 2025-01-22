import json
from collections import defaultdict
import argparse

def bounding_box_contains_point(bbox, x, y):
    return bbox['x_min'] <= x <= bbox['x_max'] and bbox['y_min'] <= y <= bbox['y_max']

def get_smallest_bounding_box(coordinate_x, coordinate_y, accessibility_tree):
    def find_smallest_bbox(tree, x, y):
        smallest_bbox = None
        for node in tree:
            if isinstance(node, dict):
                bbox = node['bbox_pixels']
                if bounding_box_contains_point(bbox, x, y):
                    if smallest_bbox is None or (
                            (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min']) < 
                            (smallest_bbox['x_max'] - smallest_bbox['x_min']) * 
                            (smallest_bbox['y_max'] - smallest_bbox['y_min'])):
                        smallest_bbox = bbox
            elif isinstance(node, list):
                child_smallest_bbox = find_smallest_bbox(node, x, y)
                if child_smallest_bbox:
                    if smallest_bbox is None or (
                            (child_smallest_bbox['x_max'] - child_smallest_bbox['x_min']) * 
                            (child_smallest_bbox['y_max'] - child_smallest_bbox['y_min']) < 
                            (smallest_bbox['x_max'] - smallest_bbox['x_min']) * 
                            (smallest_bbox['y_max'] - smallest_bbox['y_min'])):
                        smallest_bbox = child_smallest_bbox
        return smallest_bbox
    return find_smallest_bbox(accessibility_tree, coordinate_x, coordinate_y)

def find_smallest_box_pred(pred_x, pred_y, accessibility_tree):
    def find_smallest_bbox(x, y, tree):
        smallest_node = None
        smallest_area = float('inf')
        
        for node in tree:
            if isinstance(node, dict):
                bbox = node['bbox_pixels']
                if bounding_box_contains_point(bbox, x, y):
                    area = (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min'])
                    if area < smallest_area:
                        smallest_area = area
                        smallest_node = node
            elif isinstance(node, list):
                child_smallest_node = find_smallest_bbox(x, y, node)
                if child_smallest_node:
                    child_bbox = child_smallest_node['bbox_pixels']
                    child_area = (child_bbox['x_max'] - child_bbox['x_min']) * (child_bbox['y_max'] - child_bbox['y_min'])
                    if child_area < smallest_area:
                        smallest_area = child_area
                        smallest_node = child_smallest_node
        
        return smallest_node
    
    return find_smallest_bbox(pred_x, pred_y, accessibility_tree)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_file(sample_file, plan_file, ans_file):
    sample_data = load_json(sample_file)
    plan_data = load_jsonl(plan_file)
    ans_data = load_jsonl(ans_file)

    sample_index = {(entry['episode_id'], entry['step']): entry for entry in sample_data}
    ans_index = {(entry['episode_id'], entry['step']): entry for entry in ans_data}

    results = defaultdict(int)

    correct_steps = 0
    total_steps = 0
    grounding_steps = 0
    correct_grounding_steps = 0

    for plan_entry in plan_data:
        episode_id = plan_entry['episode_id']
        step = plan_entry['step']
        
        sample_entry = sample_index.get((episode_id, step))
        if not sample_entry:
            continue
        
        total_steps += 1
        gold_action = sample_entry['action']
        pred_action = plan_entry['action']
        
        if pred_action['action_type'] in ['click', 'long_press', 'type_text'] and gold_action['action_type'] == pred_action['action_type']:
            grounding_steps += 1
            ans_entry = ans_index.get((episode_id, step))
            output = ans_entry['output']
            x, y = map(float, output.strip('()').split(', '))
            scale = ans_entry['scale']
            x /= scale
            y /= scale
            bbox = get_smallest_bounding_box(gold_action['x'], gold_action['y'], sample_entry['accessibility_tree'])
            if bbox and bounding_box_contains_point(bbox, x, y):
                if gold_action['action_type'] == 'type_text':
                    if gold_action['text'] == pred_action['text']:
                        correct_steps += 1
                        correct_grounding_steps += 1
                else:
                    correct_steps += 1
                    correct_grounding_steps += 1
        else:
            if all(gold_action.get(key) == pred_action.get(key) for key in gold_action.keys()):
                correct_steps += 1
        
        # check equivalent action
        if (pred_action['action_type'] == 'click' and gold_action['action_type'] == 'open_app') or \
        (pred_action['action_type'] == 'open_app' and gold_action['action_type'] == 'click'):
            pred_x = pred_action.get('x')
            pred_y = pred_action.get('y')

            if pred_action['action_type'] == 'click':
                grounding_steps += 1
            
            ans_entry = ans_index.get((episode_id, step))
            output = ans_entry['output']
            x, y = map(float, output.strip('()').split(', '))
            scale = ans_entry['scale']
            pred_x = x / scale
            pred_y = y / scale

            element = find_smallest_box_pred(pred_x, pred_y, sample_entry['accessibility_tree'])
            if element:
                text = element.get('text', "")
                if text:
                    text = text.lower()
                content = element.get("content_description", "")
                if content:
                    content = content.lower()
                app_name = gold_action.get("app_name", "").lower()
                if (text and app_name in text) or (content and app_name in content):
                    correct_steps += 1
                    correct_grounding_steps += 1

        if (pred_action['action_type'] == 'click' and gold_action['action_type'] == 'navigate_back') or \
        (pred_action['action_type'] == 'navigate_back' and gold_action['action_type'] == 'click'):
            pred_x = pred_action.get('x')
            pred_y = pred_action.get('y')

            grounding_steps += 1
            ans_entry = ans_index.get((episode_id, step))
            output = ans_entry['output']
            x, y = map(float, output.strip('()').split(', '))
            scale = ans_entry['scale']
            pred_x = x / scale
            pred_y = y / scale
            
            element = find_smallest_box_pred(pred_x, pred_y, sample_entry['accessibility_tree'])
            if element:
                text = element.get('text', "")
                if text:
                    text = text.lower()
                content = element.get("content_description", "")
                if content:
                    content = content.lower()
                if (text and "back" in text) or (content and "back" in content):
                    correct_steps += 1
                    correct_grounding_steps += 1

    results["correct_steps"] = correct_steps
    results["total_steps"] = total_steps
    results["grounding_steps"] = grounding_steps
    results["correct_grounding_steps"] = correct_grounding_steps

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and evaluate AndroidControl task steps")
    parser.add_argument("--sample_file", type=str, required=True, help="Path to the sample JSON file")
    parser.add_argument("--plan_file", type=str, required=True, help="Path to the plan JSONL file")
    parser.add_argument("--ans_file", type=str, required=True, help="Path to the answer JSONL file")

    args = parser.parse_args()

    results = process_file(args.sample_file, args.plan_file, args.ans_file)
    
    accuracy = results['correct_steps'] / results['total_steps'] if results['total_steps'] > 0 else 0
    grounding_accuracy = results['correct_grounding_steps'] / results['grounding_steps'] if results['grounding_steps'] > 0 else 0

    print(f"Accuracy: {accuracy:.2%} ({results['correct_steps']}/{results['total_steps']})")
    print(f"Grounding Accuracy: {grounding_accuracy:.2%} ({results['correct_grounding_steps']}/{results['grounding_steps']})")
