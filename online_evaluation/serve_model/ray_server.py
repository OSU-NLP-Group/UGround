import requests
from starlette.requests import Request
from typing import Any, Dict
import torch
from ray import serve
import asyncio
import base64
import pickle
from PIL import Image
import io
from filelock import FileLock
import json

# LLaVA Dependency #
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image, ImageDraw, ImageFont

import numpy as np

import asyncio


# LLaVA Dependency #

def pre_resize_by_width(image, default_width=(1344, 672)):
    original_width, original_height = image.size
    if original_width >= original_height:
        new_width = default_width[0]
        resize_scale = new_width / original_width
        new_height = round(original_height * resize_scale)
    else:
        new_width = default_width[1]
        resize_scale = new_width / original_width
        new_height = round(original_height * resize_scale)
    resized_image = image.resize((new_width, new_height))
    return resized_image, resize_scale


def get_scale_factor(original_size):
    original_width, original_height = original_size
    new_width = min(nearest_multiple_of_224_at_least_224(original_width, ceiling=False), 1344)
    scale_factor = new_width / original_width
    return scale_factor


def nearest_multiple_of_224_at_least_224(num, ceiling=False):
    if num <= 224:
        return 224
    division, remainder = divmod(num, 224)
    if ceiling and remainder > 0:
        return (division + 1) * 224
    if remainder < 112:
        return division * 224
    else:
        return (division + 1) * 224


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
class LlavaDeployment:
    def __init__(self):
        print("A INIT")
        disable_torch_init()

    def reconfigure(self, config: Dict[str, Any]):
        print("A reconfigure")
        self.llava_uground_model_path = "/disk2/UGround"
        self.llava_uground_model_name = get_model_name_from_path(self.llava_uground_model_path)
        self.llava_uground_tokenizer, self.llava_uground_model, self.llava_uground_image_processor, self.llava_uground_context_len = load_pretrained_model(
            self.llava_uground_model_path, None, self.llava_uground_model_name)

    async def __call__(self, request: Request) -> str:
        try:
            print("A called got it")
            # return "Made a call"
            request = await request.json()
            default_width = (1344, 672)
            if "width" in request:
                default_width = request["width"]

            image = Image.open(io.BytesIO(base64.b64decode(request["image"])))
            # Resize the image
            # new img scale and process
            # kept same with https://github.com/boyugou/llava_uground/blob/main/single_infer.py
            resized_image, pre_resize_scale = pre_resize_by_width(image)
            image_tensor, image_new_size = process_images([resized_image], self.llava_uground_image_processor,
                                                          self.llava_uground_model.config)

            target_element = request["prompt"]
            qs = f"In the screenshot, where are the pixel coordinates (x, y) of the element corresponding to \"{target_element}\"?"
            cur_prompt = qs
            if self.llava_uground_model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.llava_uground_tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = self.llava_uground_model.generate(
                    input_ids,
                    # images=image_tensor.unsqueeze(0).half().cuda(),
                    images=image_tensor.half().cuda(),
                    image_sizes=[image_new_size],
                    do_sample=False,
                    temperature=0.,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=16384,
                    use_cache=True)

            outputs = self.llava_uground_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            try:
                output_coordinates = eval(outputs)
                fixed_coordinates = tuple(x / pre_resize_scale for x in output_coordinates)
            except Exception as e:
                output_coordinates = str(e)
                fixed_coordinates = str(e)
            # return str(coord)
            return json.dumps({"out": outputs, "fix_c": fixed_coordinates, "out_c": output_coordinates})
        except Exception as e:
            return json.dumps({"out": None, "fix_c": None, "out_c": str(e)})


llava = LlavaDeployment.bind()