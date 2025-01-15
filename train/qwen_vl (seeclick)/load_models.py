from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
# from peft import AutoPeftModelForCausalLM
#
# model = AutoPeftModelForCausalLM.from_pretrained(
#     '/fs/ess/PAS1576/boyu_gou/train_vlm/ui_llava_fine_tune/checkpoints/orbyai/qwen-vl-web-hy/', # path to the output directory
#     device_map="auto",
#     trust_remote_code=True
# ).eval()
#
# merged_model = model.merge_and_unload()
#
# merged_model.save_pretrained('/cpfs01/user/chengkanzhi/checkpoint_qwen/QWen-VL-Chat-UGP-8000', max_shard_size="2048MB", safe_serialization=True)
# print("Save Success")
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
