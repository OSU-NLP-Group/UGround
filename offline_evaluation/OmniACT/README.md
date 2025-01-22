# OmniACT Evaluation Pipeline
This folder contains the pipeline for evaluating grounding agents in the OmniACT benchmark. Below is an overview of the scripts, their usage, and the order in which they should be executed.

1. **`sample.py` (optional)**

   Preprocess the dataset and sample tasks.

   ```bash
   python sample.py --base_path <path_to_dataset> --test_file <path_to_test_json> --output_file <path_to_sample_json> --sample_num <num_samples>
   ```

2. **`embed_examples.py`**

   Generate in-context example embeddings.

   ```bash
   python embed_examples.py --base_path <path_to_dataset> --train_file <path_to_train_json> --output_file <path_to_output_embeddings>
   ```

3. **`gpt_plan.py`**

   Generate plan files using a GPT model and the embeddings.

   ```bash
   export OPENAI_API_KEY="Your OpenAI API Key"
   python gpt_plan.py --gpt_model <gpt_model> --embedding_file <path_to_embedding_file> --sample_path <path_to_sample_json> --base_path <path_to_dataset> --output_path <path_to_output_plan>
   ```

4. **`extract_grounding_query.py`**

   Evaluate the sequence score from the generated plan files and extract grounding queries in one step.

   ```bash
   python extract_grounding_query.py --plan_file <path_to_plan_file> --base_path <path_to_dataset> --seq_output_file <path_to_seq_score_output> --query_output_file <path_to_grounding_query_output>
   ```

5. **Grounding Model Inference**

   Perform grounding model inference using the queries generated in the previous step. 

6. **`eval_action.py`**

   Evaluate the action score based on the sequence score file and grounding results.

   ```bash
   python eval_action.py --base_path <path_to_dataset> --seq_file <path_to_seq_score_file> --ans_file <path_to_grounding_answer_file>
   ```



**`file_schemas.py`** defines the required fields for the `plan_jsonl`, `query_jsonl`, `seq_jsonl`, and `ans_jsonl` files.