import os
import json
import re

# Define the path to the directory containing the JSON files
directory_path = '/Users/geminigby/PycharmProjects/uground_new_release/online_evaluation/Mind2Web-Live-Results/4o/json_result'

# Initialize variables for statistics
total_tasks = 0
successful_tasks = 0
total_step_success = 0  # Total score of all steps
total_step_count = 0  # Total count of all steps
efficiency_scores = []
successful_task_ids = []  # List to store IDs of fully successful tasks


# Function to calculate score rate
def score_rate(score):
    first, second = score.split("/")
    return float(first) / float(second)


# Traverse the directory and process each JSON file
for filename in os.listdir(directory_path):
    if filename.endswith(".json"):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            task_id = data['id']
            total_tasks += 1

            # Calculate if all evaluate_steps scores are 1
            evaluate_steps = data.get("evaluate_steps", [])
            all_scores_one = all(step.get("score", 0) == 1 for step in evaluate_steps)
            if all_scores_one:
                successful_tasks += 1
                successful_task_ids.append(task_id)  # Store the task ID

            # Calculate step-wise score
            step_scores = sum(step.get("score", 0) for step in evaluate_steps)
            step_count = len(evaluate_steps)

            # Update total step scores and step count for MICRO calculation
            total_step_success += step_scores
            total_step_count += step_count

            # Calculate efficiency score
            reference_task_length = data.get("reference_task_length", 0)
            if step_scores > 0:
                efficiency_score = reference_task_length / step_scores
                efficiency_scores.append(efficiency_score)

# Calculate whole task success rate
whole_task_success_rate = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0

# Calculate MICRO step success rate
step_success_rate = (total_step_success / total_step_count) if total_step_count > 0 else 0

# Calculate average efficiency score
average_efficiency_score = (sum(efficiency_scores) / len(efficiency_scores)) if len(efficiency_scores) > 0 else 0

# Print the results
print("TASKS: ", total_tasks)
print(f"Whole Task Success Rate: {whole_task_success_rate:.2f}%")
print(f"Completion Rate: {step_success_rate:.4f}")
print(f"Average Efficiency Score: {average_efficiency_score:.4f}")
print("Successful Task IDs: ", successful_task_ids)
