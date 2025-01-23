import argparse
import os
import json
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from PIL import Image
import base64
from openai import AsyncOpenAI
from asyncio import Semaphore

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",  # vLLM service address
    api_key="token-abc123",  # Must match the --api-key used in vLLM serve
)

async def async_encode_image(image_path):
    """Encode image as a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def format_openai_template(description: str, base64_image):
    """Format OpenAI request template."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {
                    "type": "text",
                    "text": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {description}

Answer:""",
                },
            ],
        },
    ]

async def process_question(args, line, semaphore):
    """Process a single question."""
    async with semaphore:
        try:
            if args.image_folder:
                image_base_dir = os.path.expanduser(args.image_folder)
                image_path = os.path.join(image_base_dir, line[args.image_key])
            else:
                image_path = os.path.expanduser(line[args.image_key])

            description = line["description"]

            image = Image.open(image_path)
            image = image.convert("RGB")
            width, height = image.size

            base64_image = await async_encode_image(image_path)

            # Format request
            messages = format_openai_template(description, base64_image)

            # Call model API
            completion = await client.chat.completions.create(
                model=args.model_path,  # Use the specified model
                messages=messages,
                temperature=args.temperature,
            )

            # Parse model output
            response_text = completion.choices[0].message.content.strip()
            ratio_coords = eval(response_text)
            x_ratio, y_ratio = ratio_coords

            # Convert to absolute coordinates
            x_coord = int(x_ratio / 1000 * width)
            y_coord = int(y_ratio / 1000 * height)

            line["output"] = f"({x_coord}, {y_coord})"
            line["model_id"] = os.path.expanduser(args.model_path)
            line["scale"] = 1.0

            return line
        except Exception as e:
            print(f"Error processing question: {e}")
            return None

async def eval_model(args):
    """Evaluate the model."""
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    semaphore = Semaphore(150)  # Limit concurrency
    tasks = [process_question(args, line, semaphore) for line in questions]
    results = []

    for result in async_tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await result)

    with open(answers_file, "w") as ans_file:
        for line in results:
            ans_file.write(json.dumps(line) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="osunlp/UGround-V1-7B")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--image-key", type=str, default="img_filename")
    parser.add_argument("--temperature", type=float, default=0)
    
    args = parser.parse_args()

    asyncio.run(eval_model(args))
