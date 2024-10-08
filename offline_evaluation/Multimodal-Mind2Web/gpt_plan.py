# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import json
import io
from PIL import Image
from tqdm import tqdm
import os
import re
import base64
import openai
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the 'OPENAI_API_KEY' environment variable.")

system_prompt = '''Imagine that you are imitating humans doing web navigation for a task step by step.
At each stage, you can see the webpage like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history.
You need to decide on the first following action to take.
You can click an element with the mouse, select an option, type text with the keyboard, or scroll down.(For your understanding, they are like the click(), select_option(), type() and mouse.wheel() functions in playwright respectively.)'''

question_description = '''
The screenshot below shows the webpage you see.
First, observe the current webpage and think through your next step based on the task and previous actions.

To be successful, it is important to follow the following rules: 
1. Make sure you understand the task goal to avoid wrong actions.
2. Ensure you carefully examine the current screenshot and issue a valid action based on the observation. 
3. You should only issue one action at a time.
4. The element you want to operate with must be fully visible in the screenshot. If it is only partially visible, you need to "SCROLL DOWN" to see the entire element.
5. The necessary element to achieve the task goal may be located further down the page. If you don't want to interact with any elements, simply select "SCROLL DOWN" to move to the section below.

Explain the action you want to perform and the element you want to operate with (if applicable). Describe your thought process and reason in 3 sentences.

Finally, conclude your answer using the format below.
Ensure your answer strictly follows the format and requirements provided below, and is clear and precise. 
The action, element, and value should each be on three separate lines. '''

action_format = "ACTION: Choose an action from {CLICK, TYPE, SELECT, SCROLL DOWN}. You must choose one of this four, instead of choosing None."

element_format = '''ELEMENT: Provide a description of the element you want to operate. (If ACTION == SCROLL DOWN, this field should be None.)
It should include the element's identity, type (button, input field, dropdown menu, tab, etc.), and text on it (if have).
Ensure your description is both concise and complete, covering all the necessary information and less than 30 words.
If you find identical elements, specify its location and details to differentiate it from others.
'''

value_format = "VALUE: Provide additional input based on ACTION.\n\nThe VALUE means:\nIf ACTION == TYPE, specify the " \
               "text to be typed.\nIf ACTION == SELECT, specify the option to be chosen.\nOtherwise, write 'None'."

def encode_image(image):
    """Convert a PIL image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_gpt_response(client, model, messages):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process web navigation tasks using GPT-4V.")
    parser.add_argument('--gpt_model', type=str, default='gpt-4-turbo', help='the name of GPT model to use')
    parser.add_argument("--input_file", type=str, required=True, help="Path to sample(blocks) JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output plan JSONL file")
    parser.add_argument("--blocks", type=str, required=True, help="Directory for block images")

    args = parser.parse_args()

    input_file_path = args.input_file
    output_file_path = args.output_file
    block_image_dir = args.blocks
    gpt_model = args.gpt_model

    client = OpenAI(api_key=api_key, max_retries=5, timeout=60)

    with open(input_file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    with open(output_file_path, "w") as output_file:
        for row in tqdm(data):
            annotation_id = row['annotation_id']
            action_uid = row['action_uid']
            blocks_path = row['blocks_path']
            target_blocks = row['target_blocks']
            task_description = row['task']
            previous_actions = row.get('previous_actions', [])

            block_num = 0

            while True: 
                block_image_path = os.path.join(block_image_dir, blocks_path, f"{block_num}.png")
                if not os.path.exists(block_image_path):
                    break

                query = f"You are asked to complete the following task: {task_description}\n\n"
                
                previous_action_text = "Previous Actions:\n"
                if previous_actions is None:
                    previous_actions = []
                for action_text in previous_actions:
                    previous_action_text += action_text + "\n"
                
                query += previous_action_text + "\n" + question_description + "\n\n"
                query += action_format + "\n\n" + element_format + "\n\n" + value_format

                with Image.open(block_image_path) as image:
                    base64_image = encode_image(image)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                        ]
                    }
                ]

                response = get_gpt_response(client, gpt_model, messages)
                print('response', response)
                if response is None:
                    break

                action_match = re.search(r"ACTION:\s*(.*?)\s*ELEMENT:", response, re.DOTALL)
                action = action_match.group(1).strip() if action_match else ""

                element_match = re.search(r"ELEMENT:\s*(.*?)\s*VALUE:", response, re.DOTALL)
                element_description = element_match.group(1).strip() if element_match else ""

                value_match = re.search(r"VALUE:\s*(.*?)$", response, re.DOTALL)
                value = value_match.group(1).strip() if value_match else ""

                max_target_key = max(int(key) for key in target_blocks.keys())

                # Stop if the action is not "SCROLL DOWN", the current block exceeds the max block containing the target element,
                # or if the last block has been reached.
                if action != "SCROLL DOWN" or block_num > max_target_key or not os.path.exists(os.path.join(block_image_dir, blocks_path, f"{block_num + 1}.png")):
                    break

                block_num += 1
            
            # Save final results
            row["ans_block"] = block_num
            row["gpt_action"] = action
            row["gpt_value"] = value
            row["description"] = element_description
            row["response"] = response

            output_file.write(json.dumps(row) + "\n")
            output_file.flush()
    
    print(f'File saved at {output_file_path}')
