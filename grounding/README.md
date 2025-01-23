# UGround-V1 Inference Guidelines
As Qwen2-VL recommended, we use vLLM for fast deployment and inference.

You can find more instruction about training and inference in [Qwen2-VL's Official Repo](https://github.com/QwenLM/Qwen2-VL).

Here we use float16 instead of bfloat16 for more stable decoding (See details in [vLLM's doc](https://docs.vllm.ai/en/latest/usage/faq.html#:~:text=Mitigation Strategies))

## Installation

```sh
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install accelerate
pip install qwen-vl-utils

# Change to your CUDA version
CUDA_VERSION=cu121
pip install 'vllm==0.6.1' --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}
```

## Usage

### 1. API Service Mode

Start the vLLM service:

```sh
vllm serve <model_path: e.g. osunlp/UGround-V1-7B> --api-key token-abc123 --dtype float16
```

Run inference using the service:

```sh
python grounding/uground_qwen2vl_serve.py --model-path <model_path> --question-file <query_jsonl> --answers-file <grounding_answer_jsonl> --image-folder <screenshot_dir> --image-key <image_field_name> --temperature <temperature>
```

`<model-path>` must match the model path used when starting the vLLM service.

### 2. Local Inference Mode

Run inference locally:

```sh
python grounding/uground_qwen2vl.py --model-path <model_path> --question-file <query_jsonl> --answers-file <grounding_answer_jsonl> --image-folder <screenshot_dir> --image-key <image_field_name>
```

Optional Parameters:

- `--dtype` (default: `float16`)
- `--temperature` (default: `0`)
- `--max-num-seqs`: Maximum sequences processed simultaneously (default: `1`).

### Notes

- The **API Service Mode** generally provides better performance, and the reported scores are based on this mode.