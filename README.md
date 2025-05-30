


# UGround
This is the official code repository for the project: *Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents* [**ICLR'25 Oral**]. This work is a collaboration between [OSU NLP Group](https://x.com/osunlp) and [Orby AI](https://www.orby.ai/).
<img width="1556" alt="image" src="https://github.com/user-attachments/assets/18c6a9f4-31cc-4817-a252-bfd0dbaf3fd6">

- [🏠Homepage](https://osu-nlp-group.github.io/UGround)
- [📖Paper](https://arxiv.org/abs/2410.05243)
- [😊Model Weights](https://huggingface.co/collections/osunlp/uground-677824fc5823d21267bc9812)
- [😊Demo](https://huggingface.co/spaces/orby-osu/UGround)
- [😊Training Data](https://huggingface.co/datasets/osunlp/UGround-V1-Data)

<h3>Updates</h3>

- 2025/05/01: The bounding-box version of the training data is now available [here](https://huggingface.co/datasets/osunlp/UGround-V1-Data-Box).

- 2025/01/23: Our [training data](https://huggingface.co/datasets/osunlp/UGround-V1-Data) for the UGround-V1 series (Initial/Qwen2-VL) has been released. We also have provided a comprehensive evaluation suite packed with meaningful resources to help researchers test GUI Agents and grounding models with ease. Try them out! The performance of Qwen2-VL-based UGround-V1 on several benchmarks are also updated on the [homepage](https://osu-nlp-group.github.io/UGround) (e.g., AndroidWorld: 33->44). 

- 2025/01/05: Qwen2-VL-based UGround-V1 acheives SOTA results on a new and comprehensive GUI grounding benchmark ScreenSpot-Pro, substaintially outperforms prior models (18.9->31.1). Check the [results](https://gui-agent.github.io/grounding-leaderboard/) and [our tweet](https://x.com/BoyuGouNLP/status/1876299190889742391).

- 2025/01/03: Qwen2-VL-based UGround-V1 has been released ([2B](https://huggingface.co/osunlp/UGround-V1-2B), [7B](https://huggingface.co/osunlp/UGround-V1-7B), [72B](https://huggingface.co/osunlp/UGround-V1-72B)). Check thier performance in [Main Results](#main-results).

- 2024/10/07: Preprint is arXived. Demo is live. Code coming soon.

- 2024/08/06: Website is live. The initial manuscript and results are available.



<h3>Release Plans:</h3>

- [x] [Model Weights](https://huggingface.co/collections/osunlp/uground-677824fc5823d21267bc9812)
  - [x] Initial Version (the one used in the paper)
  - [x] Qwen2-VL-Based V1 (2B, 7B, 72B)
- [x] Code
  - [x] [Training and Inference](https://github.com/OSU-NLP-Group/UGround/tree/main/train)
  - [x] Offline Experiments (Code, Results, and Useful Resources)
    - [x] [ScreenSpot](https://github.com/OSU-NLP-Group/UGround/tree/main/offline_evaluation/ScreenSpot)
    - [x] [Multimodal-Mind2Web](https://github.com/OSU-NLP-Group/UGround/tree/main/offline_evaluation/Multimodal-Mind2Web)
    - [x] [OmniAct](https://github.com/OSU-NLP-Group/UGround/tree/main/offline_evaluation/OmniACT)
    - [x] [Android Control](https://github.com/OSU-NLP-Group/UGround/tree/main/offline_evaluation/AndroidControl)
  - [x] Online Experiments
    - [x] [Mind2Web-Live-SeeAct-V](https://github.com/boyugou/Mind2Web_Live_SeeAct_V)
    - [x] [AndroidWorld-SeeAct-V](https://github.com/boyugou/android_world_seeact_v)
  - [ ] Data Synthesis Pipeline (Coming Soon)
- [x] [Training Data (V1)](https://huggingface.co/datasets/osunlp/UGround-V1-Data)
- [x] Online Demo (HF Spaces)


# Main Results

## GUI Visual Grounding: ScreenSpot (Standard Setting)

![image](https://github.com/user-attachments/assets/d608c189-2cac-4fd9-9b25-d60847916159)

| ScreenSpot (Standard)         | Arch             | SFT data         | Mobile-Text | Mobile-Icon | Desktop-Text | Desktop-Icon | Web-Text | Web-Icon | Avg      |
| ----------------------------- | ---------------- | ---------------- | ----------- | ----------- | ------------ | ------------ | -------- | -------- | -------- |
| InternVL-2-4B                 | InternVL-2       |                  | 9.2         | 4.8         | 4.6          | 4.3          | 0.9      | 0.1      | 4.0      |
| Groma                         | Groma            |                  | 10.3        | 2.6         | 4.6          | 4.3          | 5.7      | 3.4      | 5.2      |
| Qwen-VL                       | Qwen-VL          |                  | 9.5         | 4.8         | 5.7          | 5.0          | 3.5      | 2.4      | 5.2      |
| MiniGPT-v2                    | MiniGPT-v2       |                  | 8.4         | 6.6         | 6.2          | 2.9          | 6.5      | 3.4      | 5.7      |
| GPT-4                         |                  |                  | 22.6        | 24.5        | 20.2         | 11.8         | 9.2      | 8.8      | 16.2     |
| GPT-4o                        |                  |                  | 20.2        | 24.9        | 21.1         | 23.6         | 12.2     | 7.8      | 18.3     |
| Fuyu                          | Fuyu             |                  | 41.0        | 1.3         | 33.0         | 3.6          | 33.9     | 4.4      | 19.5     |
| Qwen-GUI                      | Qwen-VL          | GUICourse        | 52.4        | 10.9        | 45.9         | 5.7          | 43.0     | 13.6     | 28.6     |
| Ferret-UI-Llama8b             | Ferret-UI        |                  | 64.5        | 32.3        | 45.9         | 11.4         | 28.3     | 11.7     | 32.3     |
| Qwen2-VL                      | Qwen2-VL         |                  | 61.3        | 39.3        | 52.0         | 45.0         | 33.0     | 21.8     | 42.1     |
| CogAgent                      | CogAgent         |                  | 67.0        | 24.0        | 74.2         | 20.0         | 70.4     | 28.6     | 47.4     |
| SeeClick                      | Qwen-VL          | SeeClick         | 78.0        | 52.0        | 72.2         | 30.0         | 55.7     | 32.5     | 53.4     |
| OS-Atlas-Base-4B              | InternVL-2       | OS-Atlas         | 85.7        | 58.5        | 72.2         | 45.7         | 82.6     | 63.1     | 68.0     |
| OmniParser                    |                  |                  | 93.9        | 57.0        | 91.3         | 63.6         | 81.3     | 51.0     | 73.0     |
| **UGround (Initial)**         | LLaVA-UGround-V1 | UGround-V1       | 82.8        | 60.3        | 82.5         | 63.6         | 80.4     | 70.4     | 73.3     |
| Iris                          | Iris             | SeeClick         | 85.3        | 64.2        | 86.7         | 57.5         | 82.6     | 71.2     | 74.6     |
| ShowUI-G                      | ShowUI           | ShowUI           | 91.6        | 69.0        | 81.8         | 59.0         | 83.0     | 65.5     | 75.0     |
| ShowUI                        | ShowUI           | ShowUI           | 92.3        | 75.5        | 76.3         | 61.1         | 81.7     | 63.6     | 75.1     |
| Molmo-7B-D                    |                  |                  | 85.4        | 69.0        | 79.4         | 70.7         | 81.3     | 65.5     | 75.2     |
| **UGround-V1-2B (Qwen2-VL)**  | Qwen2-VL         | UGround-V1       | 89.4        | 72.0        | 88.7         | 65.7         | 81.3     | 68.9     | 77.7     |
| Molmo-72B                     |                  |                  | 92.7        | 79.5        | 86.1         | 64.3         | 83.0     | 66.0     | 78.6     |
| Aguvis-G-7B                   | Qwen2-VL         | Aguvis-Stage-1   | 88.3        | 78.2        | 88.1         | 70.7         | 85.7     | 74.8     | 81.0     |
| OS-Atlas-Base-7B              | Qwen2-VL         | OS-Atlas         | 93.0        | 72.9        | 91.8         | 62.9         | 90.9     | 74.3     | 81.0     |
| Aria-UI                       | Aria             | Aria-UI          | 92.3        | 73.8        | 93.3         | 64.3         | 86.5     | 76.2     | 81.1     |
| Claude (Computer-Use)         |                  |                  | **98.2**    | **85.6**    | 79.9         | 57.1         | **92.2** | 84.5     | 82.9     |
| Aguvis-7B                     | Qwen2-VL         | Aguvis-Stage-1&2 | 95.6        | 77.7        | 93.8         | 67.1         | 88.3     | 75.2     | 83.0     |
| Project Mariner               |                  |                  |             |             |              |              |          |          | 84.0     |
| CogAgent-9B-20241220          | GLM-4V-9B        |                  |             |             |              |              |          |          | 85.4     |
| **UGround-V1-7B (Qwen2-VL)**  | Qwen2-VL         | UGround-V1       | 93.0        | 79.9        | 93.8         | 76.4         | 90.9     | 84.0     | 86.3     |
| AGUVIS-72B                    | Qwen2-VL         | Aguvis-Stage-1&2 | 94.5        | 85.2        | **95.4**     | 77.9         | 91.3     | 85.9     | 88.4     |
| **UGround-V1-72B (Qwen2-VL)** | Qwen2-VL         | UGround-V1       | 94.1        | 83.4        | 94.9         | **85.7**     | 90.4     | **87.9** | **89.4** |









## GUI Visual Grounding: ScreenSpot (Agent Setting)






| Planner | Agent-Screenspot         | arch             | SFT data         | Mobile-Text | Mobile-Icon | Desktop-Text | Desktop-Icon | Web-Text | Web-Icon | Avg  |
| ------- | ------------------------ | ---------------- | ---------------- | ----------- | ----------- | ------------ | ------------ | -------- | -------- | ---- |
| GPT-4o  | Qwen-VL                  | Qwen-VL          |                  | 21.3        | 21.4        | 18.6         | 10.7         | 9.1      | 5.8      | 14.5 |
| GPT-4o  | Qwen-GUI                 | Qwen-VL          | GUICourse        | 67.8        | 24.5        | 53.1         | 16.4         | 50.4     | 18.5     | 38.5 |
| GPT-4o  | SeeClick                 | Qwen-VL          | Web, Mobile, ... | 81.0        | 59.8        | 69.6         | 33.6         | 43.9     | 26.2     | 52.4 |
| GPT-4o  | OS-Atlas-Base-4B         | InternVL         | OS-Atlas         | 94.1        | 73.8        | 77.8         | 47.1         | 86.5     | 65.3     | 74.1 |
| GPT-4o  | UGround (Initial)        | LLaVA-UGround-V1 | UGround-V1       | 93.4        | 76.9        | 92.8         | 67.9         | 88.7     | 68.9     | 81.4 |
| GPT-4o  | UGround-V1-2B (Qwen2-VL) | Qwen2-VL         | UGround-V1       | 94.1        | 77.7        | 92.8         | 63.6         | 90.0     | 70.9     | 81.5 |
| GPT-4o  | Molmo-72B                |                  |                  | 94.1        | 79.0        | 92.3         | 70.0         | 88.7     | 67.0     | 81.9 |
| GPT-4o  | Molmo-7B-D               |                  |                  | 93.4        | 80.8        | 91.2         | 72.9         | 88.7     | 69.4     | 82.7 |
| GPT-4o  | OS-Atlas-Base-7B         | Qwen2-VL         | OS-Atlas         | 93.8        | 79.9        | 90.2         | 66.4         | 92.6     | 79.1     | 83.7 |
| GPT-4o  | UGround-V1-7B (Qwen2-VL) | Qwen2-VL         | UGround-V1       | 94.1        | 79.9        | 93.3         | 73.6         | 89.6     | 73.3     | 84.0 |
| GPT-4o  | UGround-V1-72B (Qwen2-VL)| Qwen2-VL         | UGround-V1       | 94.5        | 79.9        | 93.8         | 75.0         | 88.7     | 75.2     | 84.5 |



## Inference of Qwen2-VL-Based UGround

### Python Environment (followed from Qwen2-VL's official repo)

```bash
#inference
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install accelerate
pip install qwen-vl-utils
pip install 'vllm==0.6.1' 
```


### vLLM server

```bash
vllm serve osunlp/UGround-V1-7B  --api-key token-abc123 --dtype float16
```
or

```bash
python -m vllm.entrypoints.openai.api_server --served-model-name osunlp/UGround-V1-7B --model osunlp/UGround-V1-7B --dtype float16 
```
You can find more instruction about training and inference in [Qwen2-VL's Official Repo](https://github.com/QwenLM/Qwen2-VL).

Here we use float16 instead of bfloat16 for more stable decoding (See details in [vLLM's doc](https://docs.vllm.ai/en/latest/usage/faq.html#:~:text=Mitigation%20Strategies))

### Visual Grounding Prompt
```python
def format_openai_template(description: str, base64_image):
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

  Answer:"""
                },
            ],
        },
    ]


messages = format_openai_template(description, base64_image)

completion = await client.chat.completions.create(
    model=args.model_path,
    messages=messages,
    temperature=0  # REMEMBER to set temperature to ZERO!
# REMEMBER to set temperature to ZERO!
# REMEMBER to set temperature to ZERO!
)

# The output will be in the range of [0,1000), which is compatible with the original Qwen2-VL
# So the actual coordinates should be (x/1000*width, y/1000*height)

```


![Untitled design](https://github.com/user-attachments/assets/31758aff-7fc8-4c83-a259-86dc27a5b90a)


## Citation Information


If you find this work useful, please consider starring our repo and citing our papers: 

```
@inproceedings{gou2025uground,
title={Navigating the Digital World as Humans Do: Universal Visual Grounding for {GUI} Agents},
author={Boyu Gou and Ruohan Wang and Boyuan Zheng and Yanan Xie and Cheng Chang and Yiheng Shu and Huan Sun and Yu Su},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=kxnoqaisCT}
}

@inproceedings{zheng2024seeact,
  title={GPT-4V(ision) is a Generalist Web Agent, if Grounded},
  author={Boyuan Zheng and Boyu Gou and Jihyung Kil and Huan Sun and Yu Su},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=piecKJ2DlB},
}
```
