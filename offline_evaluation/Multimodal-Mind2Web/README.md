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
python make_blocks.py --input_file <sample_jsonl> --output_file <blocks_jsonl> --image_folder <screenshots_dir> --output_folder <blocks_dir>
```

**3. `gpt_plan.py`**

Generate plan files using GPT models.

```bash
export OPENAI_API_KEY="Your OpenAI API Key"
python gpt_plan.py --gpt_model <model_name> --input_file <blocks_jsonl> --output_file <plan_jsonl> --blocks <blocks_dir>
```

**4. `extract_grounding_query.py`**

Extract grounding queries from the plan files.

```bash
python extract_grounding_query.py --input_file <plan_jsonl> --output_file <query_jsonl> --blocks <blocks_dir>
```

**5. Grounding Model Inference**

Perform grounding model inference using the query file generated in the previous step. This step requires running the scripts in the `Grounding` folder.

**6. `eval.py`**

Evaluate the Element Accuracy, Operation F1 and Step Success Rate based on plan and grounding results.

```bash
python eval.py --sample_file <blocks_jsonl> --plan_file <plan_jsonl> --ans_file <grounding_answer_jsonl> --blocks <blocks_dir>
```