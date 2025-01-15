from typing import Optional

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch

import re

import os
import urllib
from typing import Tuple



# Model name used to load huggingface model.
QWEN2_VL_MODEL_NAME = 'Qwen/Qwen2-VL-7B-Instruct'

# This text including the newline symbols are used by Qwen2-VL.
QWEN2_VL_TEMPLATE = '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
'''

QWEN2_VL_START_TOKEN = '<|im_start|>'
QWEN2_VL_END_TOKEN = '<|im_end|>'
QWEN2_VL_SYSTEM_TOKEN = 'system'
QWEN2_VL_USER_TOKEN = 'user'
QWEN2_VL_ASSISTANT_TOKEN = 'assistant'

# Special tokens used by Qwen2-VL to indicate image position in the text.
QWEN2_VL_IMAGE_PAD_TOKEN = '<|image_pad|>'
QWEN2_VL_IMAGE_START_TOKEN = '<|vision_start|>'
QWEN2_VL_IMAGE_END_TOKEN = '<|vision_end|>'

HF_IGNORE_INDEX = -100

# Max number of pixels. Qwen2-VL will resize images and keep aspect ratio.
MAX_PIXELS = 1344 * 1344

# Good patterns for correct actions.
GOOD_PATTERNS = [
    re.compile(r'scroll\(-?[0-1](\.[0-9]+)?, ?-?[0-1](\.[0-9]+)?\)'),
    re.compile(r'fill\([\'"][0-9]+[\'"], ?(.+?)\)'),
    re.compile(r'select_option\([\'"][0-9]+[\'"], ?(.+?)\)'),
    re.compile(r'click\([\'"][0-9]+[\'"]\)'),
    re.compile(r'mouse_click\(0\.[0-9]+, ?0\.[0-9]+\)'),
    re.compile(r'keyboard_insert_text\((.+?)\)'),
    re.compile(r'keyboard_press\((.+?)\)'),
    re.compile(r'noop\(\)')
]

# Patterns to match problematic actions if they don't match any good pattern.
_P_SCROLL_1 = re.compile(r'scroll\(([\-.0-9]+)\)')
_P_SCROLL_2 = re.compile(r'scroll\(([\-.0-9]+), ([\-.0-9]+)\)')
_P_FILL = re.compile(r'fill\((.+?)\)')
_P_KEYBOARD_TYPE = re.compile(r'keyboard_type\((.+?)\)')
_P_SEND_MSG = re.compile(r'send_msg_to_user\((.+?)\)')

# Single and double quotes.
_QUOTES = ['"', "'"]


def save_model(local_model_path="/fs/ess/PAS1576/boyu_gou/models/UGround_Qwen2_VL/ug-v1-72b-lora/ep0-ba10000-rank0.pt", model_name:Optional[str] = None, processor_name="Qwen/Qwen2-VL-72B-Instruct", merged_model_path="UGround-V1-72b"):
    if processor_name:
        print(f"Processor: {processor_name}")


    if merged_model_path:
        from peft import PeftModel
        print(f'Merging model weights...')
        if processor_name in ['Qwen/Qwen2-VL-7B-Instruct', 'Qwen/Qwen2-VL-72B-Instruct']:
            model = Qwen2VLForConditionalGeneration.from_pretrained(processor_name, device_map="auto", torch_dtype=torch.float16)
            lora_model = PeftModel.from_pretrained(model, local_model_path)
            lora_model= lora_model.merge_and_unload()
            lora_model.save_pretrained(merged_model_path)
        else:
            print(f"{processor_name} LoRA weights merging is not supported.")
        local_model_path = merged_model_path

    if processor_name:
        print('Writing preprocessor config...')
        kwargs = {}
        if processor_name in ['Qwen/Qwen2-VL-7B-Instruct', 'Qwen/Qwen2-VL-72B-Instruct']:
            # Use the same max_pixels value as training for qwen2-vl models.
            kwargs['max_pixels'] = 1344*1344
        processor = AutoProcessor.from_pretrained(processor_name, **kwargs)
        processor.save_pretrained(local_model_path)


if __name__ == "__main__":
    save_model()
