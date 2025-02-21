import json
import random
import argparse

def sample_data(input_file, output_file, sample_size=100):
    # Read the entire JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # Ensure data is a list
    if not isinstance(data, list):
        raise ValueError("Input JSON should contain a list of objects")
    
    # Sample the data
    if len(data) <= sample_size:
        sampled_data = data
        print(f"Warning: Input file contains fewer than {sample_size} entries. Using all available data.")
    else:
        sampled_data = random.sample(data, sample_size)
    
    # Write the sampled data to the output file
    with open(output_file, 'w') as outfile:
        json.dump(sampled_data, outfile, indent=2)
    
    print(f"Sampled {len(sampled_data)} data points have been written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to the full test steps JSON file")
    parser.add_argument("--output_file", help="Path to the output JSON file for sampled data")
    parser.add_argument("-n", "--num_samples", type=int, default=100, help="Number of samples to take (default: 100)")
    
    args = parser.parse_args()
    
    sample_data(args.input_file, args.output_file, args.num_samples)