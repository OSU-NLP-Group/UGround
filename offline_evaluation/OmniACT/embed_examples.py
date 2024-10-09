import json
import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
import argparse

def generate_embedding(text, tokenizer, model):
    """Generate embedding for text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for OmniACT tasks")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to the OmniACT dataset")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the OmniACT train JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output embeddings file(JSONL)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    model = AutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

    with open(args.train_file, 'r') as file:
        data = json.load(file)

    with open(args.output_file, 'w') as output_file:
        for key, value in tqdm(data.items()):
            task_file_path = os.path.join(args.base_path, value['task'])
            id = key

            with open(task_file_path, 'r') as task_file:
                lines = task_file.readlines()
                task = lines[0].strip().replace('Task: ', '')

            embedding = generate_embedding(task, tokenizer, model)

            result = {
                'id': id,
                'task': task_file_path,
                'embedding': embedding
            }

            output_file.write(json.dumps(result) + '\n')
            output_file.flush()

    print(f"Embeddings successfully saved to {args.output_file}")
