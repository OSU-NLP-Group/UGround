import json
import os
import re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse

def get_high_contrast_color(image, x, y):
    try:
        pixel = image.getpixel((x, y))
        red_brightness = pixel[0]
        blue_brightness = pixel[2]
    except:
        red_brightness = 0
        blue_brightness = 0

    if red_brightness > blue_brightness:
        return "blue"
    else:
        return "red"

def draw_multiline_text(draw, text, position, font, max_width, fill):
    lines = []
    words = text.split()
    while words:
        line = ''
        while words:
            test_line = line + words[0] + ' '
            bbox = font.getbbox(test_line)
            if bbox[2] <= max_width:
                line = test_line
                words.pop(0)
            else:
                break
        if line.strip():
            lines.append(line.strip())
        else:
            lines.append(words.pop(0))
    
    y = position[1]
    for line in lines:
        draw.text((position[0], y), line, font=font, fill=fill)
        y += font.getbbox(line)[3] - font.getbbox(line)[1]

def extract_coordinates(operation, image_path):
    tap_match = re.search(r'tap\s*\[\[(\d+),(\d+)\]\]', operation, re.IGNORECASE)
    box_match = re.search(r'\[\[(\d+),(\d+),(\d+),(\d+)\]\]', operation)

    image = Image.open(image_path)
    width, height = image.size
    
    if tap_match:
        x, y = map(int, tap_match.groups())
        x = int(width * (x / 1000))
        y = int(height * (y / 1000))
        return x, y
    elif box_match:
        x1, y1, x2, y2 = map(int, box_match.groups())
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_x = int(width * (center_x / 1000))
        center_y = int(height * (center_y / 1000))
        return center_x, center_y
    else:
        return 0, 0

def is_output_in_bbox(data, image_dir):
    bbox = data['bbox']
    output = data.get('output', "")
    image_path = os.path.join(image_dir, data["img_filename"])
    
    if not output:
        operation = data.get('operation', "")
        if operation:
            x, y = extract_coordinates(data['operation'], image_path)
            if x == 0 and y == 0:
                return False, None
        else:
            return False, None
    else:
        x, y = map(int, output.strip('()').split(', '))

        scale = data.get('scale', 1)
        x = x / scale
        y = y / scale

    bbox_x, bbox_y, bbox_width, bbox_height = bbox

    is_in_bbox = bbox_x <= x <= bbox_x + bbox_width and bbox_y <= y <= bbox_y + bbox_height
    return is_in_bbox, (x, y)

def calculate_accuracy(file_path, image_dir):
    categories = {
        'mobile_text': {'total': 0, 'correct': 0},
        'mobile_icon': {'total': 0, 'correct': 0},
        'desktop_text': {'total': 0, 'correct': 0},
        'desktop_icon': {'total': 0, 'correct': 0},
        'web_text': {'total': 0, 'correct': 0},
        'web_icon': {'total': 0, 'correct': 0},
    }
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            key = f"{data['platform']}_{data['data_type']}"
            if key in categories:
                categories[key]['total'] += 1
                correct, coords = is_output_in_bbox(data, image_dir)
                if correct:
                    categories[key]['correct'] += 1
    
    accuracies = {}
    for key, values in categories.items():
        total = values['total']
        correct = values['correct']
        accuracies[key] = correct / total if total > 0 else 0
    
    average_accuracy = sum(accuracies.values()) / len(accuracies) if accuracies else 0
    
    return accuracies, average_accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy of ScreenSpot results")
    parser.add_argument('--ans_file', type=str, required=True, help="Path to the answer file to evaluate")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing original ScreenSpot images")
    args = parser.parse_args()

    print(f"\nEvaluating: {args.ans_file}")
    accuracies, average_accuracy = calculate_accuracy(args.ans_file, args.image_dir)

    for category, accuracy in accuracies.items():
        print(f"{category}: {accuracy:.2%}")

    print(f"Average Accuracy: {average_accuracy:.2%}")

if __name__ == "__main__":
    main()