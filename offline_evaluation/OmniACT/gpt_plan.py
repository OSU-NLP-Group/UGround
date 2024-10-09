import base64
import requests
import json
import torch
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import time
import openai
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the 'OPENAI_API_KEY' environment variable.")

instruction = '''You are an excellent robotic process automation agent who needs to generate a PyAutoGUI script for the tasks given to you.
You will receive some examples to help with the format of the script that needs to be generated.

There are some actions that require you to provide an element description for the elements you want to operate on. For the description, follow the requirements below:
Element Description Requirements:
Provide a concise description of the element you want to operate. 
It should include the element's identity, type (button, input field, dropdown menu, tab, etc.), and text on it (if have).
If you find identical elements, specify its location and details to differentiate it from others.
Ensure your description is both concise and complete, covering all the necessary information and less than 30 words, and organize it into one sentence.

[IMPORTANT!!] Stick to the format of the output scripts in the example.
[IMPORTANT!!] Use only the functions from the API docs.
[IMPORTANT!!] Follow the output format strictly. Only write the script and nothing else.'''

api_reference = '''
Here is the API reference for generating the script:
def click(element=description):
    """Moves the mouse to the element corresponding to the description and performs a left click.
    Example:
    High Level Goal: Click at the rectangular red button labeled \"Next\".
    Python script:
    import pyautogui
    pyautogui.click("Rectangular red button labeled \"Next\"")
    """
    pass

def rightClick(element=description):
    """Moves the mouse to the element corresponding to the description and performs a right click.
    Example:
    High Level Goal: Right-click at link labeled \"vacation rentals\" under the \"housing\" section.
    Python script:
    import pyautogui
    pyautogui.rightClick("Link labeled \"vacation rentals\" under the \"housing\" section.")
    """
    pass

def doubleClick(element=description):
    """Moves the mouse to the element corresponding to the description and performs a double click.
    Example:
    High Level Goal: Double-click at folder named \"courses\".
    Python script:
    import pyautogui
    pyautogui.doubleClick("Folder named \"courses\"")
    """
    pass

def scroll(clicks=amount_to_scroll):
    """Scrolls the window that has the mouse pointer by float value (amount_to_scroll).
    Example:
    High Level Goal: Scroll screen by 30.
    Python script:
    import pyautogui
    pyautogui.scroll(30)
    """
    pass

def hscroll(clicks=amount_to_scroll):
    """Scrolls the window that has the mouse pointer horizontally by float value (amount_to_scroll).
    Example:
    High Level Goal: Scroll screen horizontally by 30.
    Python script:
    import pyautogui
    pyautogui.hscroll(30)
    """
    pass

def dragTo(element=description, button=holdButton):
    """Drags the mouse to the element corresponding to the description with (holdButton) pressed. holdButton can be 'left', 'middle', or 'right'.
    Example:
    High Level Goal: Drag the screen from the current position to recycle bin with the left click of the mouse.
    Python script:
    import pyautogui
    pyautogui.dragTo("Recycle bin with trash can shape", 'left')
    """
    pass

def moveTo(element = description):
    """Takes the mouse pointer to the element corresponding to the description.
    Example:
    High Level Goal: Hover the mouse pointer to search button.
    Python script:
    import pyautogui
    pyautogui.moveTo("\"Request appointment\" button")
    """
    pass

def write(str=stringType, interval=secs_between_keys):
    """Writes the string wherever the keyboard cursor is at the function calling time with (secs_between_keys) seconds between characters.
    Example:
    High Level Goal: Write "Hello world" with 0.1 seconds rate.
    Python script:
    import pyautogui
    pyautogui.write("Hello world", 0.1)
    """
    pass

def press(str=string_to_type):
    """Simulates pressing a key down and then releasing it up. Sample keys include 'enter', 'shift', arrow keys, 'f1'.
    Example:
    High Level Goal: Press the enter key now.
    Python script:
    import pyautogui
    pyautogui.press("enter")
    """
    pass

def hotkey(*args = list_of_hotkey):
    """Keyboard hotkeys like Ctrl-S or Ctrl-Shift-1 can be done by passing a list of key names to hotkey(). Multiple keys can be pressed together with a hotkey.
    Example:
    High Level Goal: Use Ctrl and V to paste from clipboard.
    Python script:
    import pyautogui
    pyautogui.hotkey("ctrl", "v")
    """
    pass
'''

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Generate text embedding
def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Load stored embeddings
def load_embeddings(file_path):
    embeddings = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            embeddings.append({
                'id': item['id'],
                'task': item['task'],
                'embedding': item['embedding']
            })
    return embeddings

# Find most similar embeddings
def find_most_similar(new_embedding, embeddings, top_k=5):
    new_embedding = np.array(new_embedding).reshape(1, -1)
    all_embeddings = np.array([e['embedding'] for e in embeddings])
    similarities = cosine_similarity(new_embedding, all_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_k]
    return [embeddings[i] for i in top_indices]

# Combine task prompt and examples for the final prompt
def create_prompt(task, examples, instruction, api_reference):
    examples_str = '''Here are some examples similar to the tasks you need to complete.
However, these examples use coordinate format for actions like click, rightClick, doubleClick, moveTo, dragTo, instead of element description.
You should only refer to the actions in these examples, and for the output format, stick to the content in the API reference.
For example, do not output pyautogui.click(100,200), instead output pyautogui.click("Gray 'Tools' menu button with a downward arrow, located in the top right corner.").
Omit 'import pyautogui', do not include any comments or thoughts. Your output should only contain the script itself.\n\n'''

    examples_str += "\n\n".join([f"Example {i+1}:\n{example}" for i, example in enumerate(examples)])
    task_prompt = f'''Based on the screenshot, generate the PyAutoGUI script for the following task: {task}
You should list all necessary steps to finish the task, which could involve multiple steps. Also ensure simplifying your steps as much as possible, avoid dividing a single task into multiple steps if it can be completed in one.'''
    prompt = instruction + "\n" + api_reference + "\n" + examples_str + "\n\n" + task_prompt
    
    return prompt

def get_gpt_response(client, model, messages):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=300
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OmniACT script')
    parser.add_argument('--gpt_model', type=str, default='gpt-4-turbo', help='GPT model name to use')
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to the embedding file')
    parser.add_argument('--sample_path', type=str, required=True, help='Path to the sample JSON file')
    parser.add_argument('--base_path', type=str, required=True, help='Base directory for OmniACT dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    model = AutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

    embeddings = load_embeddings(args.embedding_file)
    
    gpt_model = args.gpt_model
    client = OpenAI(api_key=api_key, max_retries=5, timeout=60)
    
    with open(args.sample_path, 'r') as file:
        data = json.load(file)

    with open(args.output_path, 'w') as output_file:
        for key, value in tqdm(data.items()):
            task_file_path = os.path.join(args.base_path, value['task'])
            image_path = os.path.join(args.base_path, value['image'])
            box_path = os.path.join(args.base_path, value['box'])
            id = key

            with open(task_file_path, 'r') as task_file:
                lines = task_file.readlines()
                task = lines[0].strip().replace('Task: ', '')
                output_script = ''.join(lines[1:]).strip()

            base64_image = encode_image(image_path)

            new_embedding = generate_embedding(task, tokenizer, model)
            most_similar_examples = find_most_similar(new_embedding, embeddings, top_k=5)

            examples = []
            example_ids = []
            for example in most_similar_examples:
                example_task_file_path = example['task']
                with open(example_task_file_path, 'r') as example_task_file:
                    example_lines = example_task_file.readlines()
                    example_output_script = ''.join(example_lines[1:]).strip()
                    examples.append(example_output_script)
                example_ids.append(example['id'])

            prompt = create_prompt(task, examples, instruction, api_reference)

            messages = [
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                    ]
                }
            ]

            gpt_output = get_gpt_response(client, gpt_model, messages)
            print(gpt_output, "\n")

            if gpt_output is None:
                continue

            result_line = json.dumps({
                "id": id,
                "task": value['task'],
                "image": value['image'],
                "box": value['box'],
                "gpt_output": gpt_output,
                "example_ids": example_ids,
            })

            output_file.write(result_line + '\n')
            output_file.flush()
