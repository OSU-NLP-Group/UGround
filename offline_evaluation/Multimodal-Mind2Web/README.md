# Multimodal-Mind2Web Evaluation Pipeline

This folder contains the evaluation pipeline for the **Multimodal-Mind2Web** benchmark. Below is a detailed description of the scripts, their usage, and the order in which they should be executed.

**1. `sample.py`**

Preprocess the dataset and sample tasks.

```bash
python sample.py --input_dir <dataset_dir> --output_dir <sample_dir> --samples_per_split <num_samples>
```

**2. `make_blocks.py`**

Split full screenshots into blocks.

```bash
python make_blocks.py --input_file <sample_jsonl> --output_file <sample_blocks_jsonl> --image_folder <screenshots_dir> --output_folder <blocks_dir>
```

The `sample_blocks.jsonl` files generated with the parameters `output_size = [1280, 1000]` and `padding = 200` (the parameters used in our paper) can be found in the `data/samples/` folder.

**3. `gpt_plan.py`**

Generate plan files using GPT models.

```bash
export OPENAI_API_KEY="Your OpenAI API Key"
python gpt_plan.py --gpt_model <model_name> --input_file <sample_blocks_jsonl> --output_file <plan_jsonl> --blocks <blocks_dir>
```

The plan results generated by GPT from the above `sample_blocks` file can be found in `data/{gpt_model}_results/cross_{split}_plan.jsonl`.

- `gpt_model` can be "gpt-4o" or "gpt-4-turbo"
- `split` can be "domain", "task", or "website"

**4. `extract_grounding_query.py`**

Extract grounding queries from the plan files.

```bash
python extract_grounding_query.py --input_file <plan_jsonl> --output_file <query_jsonl> --blocks <blocks_dir>
```

The queries extracted from the plan files are located in `data/{gpt_model}_results/cross_{split}_query.jsonl`.

**5. Grounding Model Inference**

Perform grounding model inference using the query file generated in the previous step. If you want to test with UGround-V1, you can use the scripts provided in the `grounding` folder.

**6. `eval.py`**

Evaluate the Element Accuracy, Operation F1 and Step Success Rate based on plan and grounding results.

```bash
python eval.py --sample_file <blocks_jsonl> --plan_file <plan_jsonl> --ans_file <grounding_answer_jsonl> --blocks <blocks_dir>
```