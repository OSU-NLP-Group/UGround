import argparse
import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm
import shortuuid
from PIL import Image

def eval_model(args):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # Read the questions and prepare the answer file
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        image_base_dir = os.path.expanduser(args.image_folder)
        image_path = os.path.join(image_base_dir, line[args.image_key])

        description = line["description"]

        # Load the image and get its dimensions
        image = Image.open(image_path)
        width, height = image.size

        # Create the prompt for the model
        prompt = f"In this UI screenshot, what is the position of the element corresponding to the command \"{description}\" (with point)?"
        
        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt},
        ])

        # Get the model's response
        with torch.no_grad():
            response, history = model.chat(tokenizer, query=query, history=None)
        
        # Parse the response to get the ratio coordinates
        ratio_coords = eval(response.strip())
        x_ratio, y_ratio = ratio_coords
        x_coord = int(x_ratio * width)
        y_coord = int(y_ratio * height)

        ans_id = shortuuid.uuid()
        line["output"] = f"({x_coord}, {y_coord})"
        line["answer_id"] = ans_id
        line["model_id"] = os.path.expanduser(args.model_path)
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
