import argparse 
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def scale_image(image_path, scale_factor):
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    print('new size', new_width, new_height)
    scaled_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    return scaled_image

def eval_model(args):
    # Load model and tokenizer
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Read questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # Prepare answer file output
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        image_base_dir = os.path.expanduser(args.image_folder)
        print('image_base_dir', image_base_dir)

        image_path = os.path.join(image_base_dir, line[args.image_key])  # Ensure this key corresponds to the image path
        print('image_path', image_path)

        description = line["description"]
        scale_factor = line["scale"]
        print('scale_factor', scale_factor)
        print('description', description)

        # Construct question
        qs = f"In the screenshot, where are the pixel coordinates (x, y) of the element corresponding to \"{description}\"?"

        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        scaled_image = scale_image(image_path, scale_factor)
        print('scaled_image size:', scaled_image.size)
        
        # Process image for model input
        image_tensor, image_new_size = process_images([scaled_image.convert('RGB')], image_processor, model.config)
        print('image_new_size', image_new_size)

        print('temperature', args.temperature)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                image_sizes=[image_new_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=16384,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print('outputs', outputs)

        ans_id = shortuuid.uuid()

        line["output"] = outputs
        line["answer_id"] = ans_id
        line["model_id"] = model_name
        ans_file.write(json.dumps(line) + "\n")
        ans_file.flush()
    
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--question-file", type=str, required=True, help="Path to the file containing questions")
    parser.add_argument("--answers-file", type=str, required=True, help="Path to save the answers")
    parser.add_argument("--image-folder", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--model-base", type=str, default=None, help="Base model path if needed")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation mode")
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of chunks to split the data into")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Index of the current chunk to process")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--image-key", type=str, default="image", help="Key in the data that contains image filename")
    args = parser.parse_args()

    eval_model(args)