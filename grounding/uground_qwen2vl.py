import argparse
import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import logging

def prepare_prompt(question, processor, args):
    """Prepare the prompt for a single query."""
    try:
        image_base_dir = os.path.expanduser(args.image_folder)
        image_path = os.path.join(image_base_dir, question[args.image_key])
        description = question["description"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
Description: {description}
Answer:"""},
                ],
            },
        ]

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        return {
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs} if image_inputs is not None else {},
            "metadata": {"image_path": image_path, "original_data": question}
        }
    except Exception as e:
        print(f"Error preparing prompt: {e}")
        return None

def eval_model(args):
    """Evaluate the model."""
    processor = AutoProcessor.from_pretrained(args.model_path)
    questions = [json.loads(q) for q in open(args.question_file, "r")]
    
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        max_num_seqs=args.max_num_seqs
    )
    
    sampling_params = SamplingParams(temperature=args.temperature)
    
    prompts_data = []
    for question in tqdm(questions, desc="Preparing prompts"):
        prompt_data = prepare_prompt(question, processor, args)
        if prompt_data:
            prompts_data.append(prompt_data)
    
    outputs = llm.generate(prompts_data, sampling_params)
    
    for output, prompt_data in zip(outputs, prompts_data):
        try:
            generated_text = output.outputs[0].text.strip()
            ratio_coords = eval(generated_text)
            x_ratio, y_ratio = ratio_coords

            with Image.open(prompt_data["metadata"]["image_path"]) as img:
                width, height = img.size

            x_abs = int(x_ratio / 1000 * width)
            y_abs = int(y_ratio / 1000 * height)

            result = dict(prompt_data["metadata"]["original_data"])
            result.update({
                "output": f"({x_abs}, {y_abs})",
                "model_id": args.model_path,
                "scale": 1.0
            })

            with open(args.answers_file, "a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            print(f"Error processing output: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="osunlp/UGround-V1-7B")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--image-key", type=str, default="img_filename")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    if os.path.exists(args.answers_file):
        os.remove(args.answers_file)

    eval_model(args)
