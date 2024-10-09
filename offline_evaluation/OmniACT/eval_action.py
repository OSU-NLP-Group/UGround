import json
from math import sqrt
import re
from nltk.translate.bleu_score import sentence_bleu
import os
import argparse

total_ideal_score = 0
sequence_match = 0
action_score = 0
total_click_penalty = 0 
total_press_penalty = 0
total_write_penalty = 0
total_penalty = 0

def get_bounds(box, cx, cy):
    for i in box:
        tl = box[i]["top_left"]
        br = box[i]["bottom_right"]
        if (tl[0] + br[0]) / 2 == cx and (tl[1] + br[1]) / 2 == cy:
            return tl, br
    return None, None

def dynamic_dirichlet_l2_penalty(tl, br, px, py):
    len_x = br[0] - tl[0]
    len_y = br[1] - tl[1]
    cx = tl[0] + len_x / 2
    cy = tl[1] + len_y / 2
    dx = abs(cx - px) - (len_x * 0.5)
    dy = abs(cy - py) - (len_y * 0.5)
    dist = sqrt((dx * (dx > 0)) ** 2 + (dy * (dy > 0)) ** 2)
    mu = sqrt(len_x ** 2 + len_y ** 2)
    score = mu / (dist + mu)
    penalty = 1 - score
    return penalty

def process_files(base_path, seq_file, ans_file):
    global total_ideal_score, sequence_match, action_score, total_click_penalty, total_press_penalty, total_write_penalty, total_penalty

    coordinates = {}
    with open(ans_file, 'r') as f:
        for line in f:
            coord = json.loads(line)
            description = coord['description']

            if description:
                scale = coord.get('scale', 1.0)
                if coord['id'] not in coordinates:
                    coordinates[coord['id']] = []
                coordinates[coord['id']].append({'description': description, 'coords': (coord['output'], scale)})

    seq_records = []
    with open(seq_file, 'r') as f:
        for line in f:
            seq_record = json.loads(line)
            task_id = seq_record['id']

            if task_id in coordinates:
                final_script = seq_record['gpt_output']
                for coord_info in coordinates[task_id]:
                    original_output, scale = coord_info['coords']
                    coords_list = original_output.strip('()').split(',')
                    scaled_x = int(float(coords_list[0]) / scale)
                    scaled_y = int(float(coords_list[1]) / scale)
                    scaled_output = f"{scaled_x},{scaled_y}"
                    final_script = final_script.replace(f"\"{coord_info['description']}\"", scaled_output)
                seq_record['final_script'] = final_script
                seq_records.append(seq_record)
            elif seq_record["seq_score"] > 0:
                seq_record['final_script'] = seq_record['gpt_output']
                seq_records.append(seq_record)
            else:
                seq_records.append(seq_record)

    for record in seq_records:
        task_path = os.path.join(base_path, record['task'])
        gpt_output = record['gpt_output']
        seq_score = record["seq_score"]
        ideal_score = record["ideal_score"]

        gold_script = open(task_path).read().strip().split('\n')[2:]
        gold_script = [line.lower() for line in gold_script]

        sample_weight = (len(gold_script)-0.9)

        total_ideal_score += ideal_score
        sequence_match += seq_score

        correct_gold_script = []

        for gold_line in gold_script:
            try:
                action_type = gold_line.split("pyautogui.")[1].split("(")[0]
            except:
                continue
            if action_type == 'click' or action_type == 'rightClick' or action_type == 'moveTo' or action_type == 'dragTo':
                max_click_penalty = sample_weight/len(gold_script)
            if action_type == 'press' or action_type == 'hotkey':
                max_press_penalty = sample_weight/len(gold_script)
            if action_type == 'write':
                max_write_penalty = sample_weight/len(gold_script)
            correct_gold_script.append(gold_line)
        
        gold_script = correct_gold_script

        if seq_score == 0:            
            continue

        llm_script = [x for x in record["final_script"].split('\n') if x.strip().startswith('pyautogui')]
        gpt_output = [x for x in record["gpt_output"].split('\n') if x.strip().startswith('pyautogui')]
        
        box_path = os.path.join(base_path, record["box"])
        box = json.load(open(box_path))

        click_penalty = 0
        press_penalty = 0
        write_penalty = 0

        for i in range(len(gold_script)):
            gold_line = gold_script[i].strip()
            gold_action = gold_line.split('pyautogui.')[1].split('(')[0]
            pred_line = llm_script[i]
            pred_action = pred_line.split('pyautogui.')[1].split('(')[0]
            desc_line = gpt_output[i]

            if gold_action in ['click', 'rightClick', 'doubleClick', 'moveTo', 'dragTo']:
                gold_cx = gold_line.split("pyautogui.")[1].split('(')[1].split(',')[0]
                gold_cy = gold_line.split("pyautogui.")[1].split('(')[1].split(',')[1].split(')')[0]
                tl, br = get_bounds(box, float(gold_cx), float(gold_cy))
                if tl == None and br == None:
                    continue

                try:
                    pred_cx = pred_line.split("pyautogui.")[1].split('(')[1].split(',')[0]
                    pred_cy = pred_line.split("pyautogui.")[1].split('(')[1].split(',')[1].split(')')[0]
                except:
                    print('coordinates not legal, id:', record['id'])
                    pred_cx = 0
                    pred_cy = 0

                try:
                    pred_cx, pred_cy = float(pred_cx), float(pred_cy)
                except:
                    try:
                        pred_cy = pred_cy.split()[0]
                        pred_cy = pred_cy.split('\'')[0]
                        pred_cx, pred_cy = float(pred_cx), float(pred_cy)
                    except:
                        pred_cx, pred_cy = 0, 0
                
                cur_penalty = dynamic_dirichlet_l2_penalty(tl, br, pred_cx, pred_cy)
                click_penalty += (1.0 / len(gold_script)) * cur_penalty

            if gold_action == 'press':
                gold_key = gold_line.split("\"")[1]
                pred_key = re.split("\"|'", pred_line)[1]
                if gold_key.strip() != pred_key.strip():
                    press_penalty += 1 / len(gold_script)
            
            if gold_action == 'hotkey':
                gold_keys = gold_line.split("(")[1].split(")")[0].split(",")
                pred_keys = pred_line.split("(")[1].split(")")[0].split(",")
                
                gold_key_set = set([x[1:-1] for x in gold_keys if len(x)>2])
                pred_key_set = set([x[1:-1] for x in pred_keys if len(x)>2])
                if gold_key_set != pred_key_set:
                    press_penalty += 1/len(gold_script)

            if gold_action == 'write':
                reference = [gold_line.split("\"")[1]]
                candidate = re.split("\"|'", pred_line)[1]
                write_penalty += (1 - sentence_bleu(reference, candidate, weights=(0.5, 0.5))) / len(gold_script)

        seq_match_flag = 1 if record["seq_score"] > 0 else 0
        action_score += (max(seq_match_flag - click_penalty - press_penalty - write_penalty, 0)) * sample_weight

        if seq_match_flag:
            total_click_penalty += click_penalty * sample_weight
            total_press_penalty += press_penalty * sample_weight
            total_write_penalty += write_penalty * sample_weight
            total_penalty += (click_penalty + press_penalty + write_penalty) * sample_weight

def main():
    parser = argparse.ArgumentParser(description="Process sequence matching and calculate penalties.")
    parser.add_argument('--base_path', type=str, required=True, help='Base directory for OmniACT dataset')
    parser.add_argument('--seq_file', type=str, required=True, help="Path to the sequence score JSONL file.")
    parser.add_argument('--ans_file', type=str, required=True, help="Path to the answer JSONL file.")
    args = parser.parse_args()

    process_files(args.base_path, args.seq_file, args.ans_file)

    print(f"Ideal score: {total_ideal_score}")
    print(f"Sequence match: {sequence_match / total_ideal_score:.3%}")
    print(f"Action match: {action_score / total_ideal_score:.3%}")
    print(f"Total click penalty: {total_click_penalty / total_ideal_score:.3%}")
    print(f"Total write penalty: {total_write_penalty / total_ideal_score:.3%}")
    print(f"Total press penalty: {total_press_penalty / total_ideal_score:.3%}")
    print(f"Total penalty: {total_penalty / total_ideal_score:.3%}")

if __name__ == "__main__":
    main()
