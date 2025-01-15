from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    'qwen-vl-web-hy/', # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()

merged_model.save_pretrained('merged_UGround_Qwen-web-hy/', max_shard_size="4096MB", safe_serialization=True)
print("Save Success")