import argparse
import torch
import os
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import shortuuid
from PIL import Image

def infer_single(model, tokenizer, device, image_path, prompt, torch_type=torch.float16):
    # Prepare the query
    query = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {prompt} ASSISTANT:"

    # Prepare the image
    image = Image.open(image_path).convert('RGB')

    # Process the input
    input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
        'images': [[input_by_model['images'][0].to(device).to(torch_type)]],
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(device).to(torch_type)]]

    # Generate the output
    gen_kwargs = {"max_length": 2048, "do_sample": False}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        output_text = tokenizer.decode(outputs[0])
        output_text = output_text.split("</s>")[0]
    
    return output_text

def extract_coordinates(operation, image_path):
    # Match the operation string for tap and box coordinates
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

def eval_model(args):
    # Load model and tokenizer
    model_path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_type = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()

    # Load questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    image_base_dir = os.path.expanduser(args.image_folder)

    for line in tqdm(questions):
        image_path = os.path.join(image_base_dir, line[args.image_key])
        
        description = line['description']

        prompt = f"What steps do I need to take to tap on \"{description}\"? (with grounding)"

        response = infer_single(model, tokenizer, device, image_path, prompt)

        # Extract coordinates from the model response
        x, y = extract_coordinates(response, image_path)

        ans_id = shortuuid.uuid()

        line["output"] = f"({x}, {y})"
        line["answer_id"] = ans_id
        line["scale"] = 1.0

        ans_file.write(json.dumps(line) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--question-file", type=str, required=True, help="Path to the file containing questions")
    parser.add_argument("--answers-file", type=str, required=True, help="Path to save the answers")
    parser.add_argument("--image-folder", type=str, required=True, help="Directory containing images")
    parser.add_argument("--image-key", type=str, default="image", help="Key in the JSON data that points to the image filename")
    args = parser.parse_args()

    eval_model(args)
