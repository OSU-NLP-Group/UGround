import json

filter_count = 0

def filter_out_bad_datapoints(file_path, output_file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    filtered_data = []

    # Iterate through each entry in the JSON
    for entry in data:
        conv_id = entry.get('id', 'unknown')
        conversations = json.loads(entry.get('conversations', []))

        # Check if any conversation in the data point has unmatched <img> tags
        has_unmatched_img_tag = False

        if len(conversations) > 30:
            conversations = conversations[:30]

        if len(str(json.dumps(conversations)))>10e3:
            continue

        first_one=True

        image_start_tag = '<img>'
        image_end_tag = '</img>'
        image_pad_tag = '<imgpad>'
        ref_start_tag = '<ref>'
        ref_end_tag = '</ref>'
        box_start_tag = '<box>'
        box_end_tag = '</box>'
        quad_start_tag = '<quad>'
        quad_end_tag = '</quad>'

        if conversations:
            first_value = conversations[0].get('value', '')
            if (
                    first_value.count(image_start_tag) >= 2 or
                    image_pad_tag in first_value or
                    ref_start_tag in first_value or
                    ref_end_tag in first_value or
                    box_start_tag in first_value or
                    box_end_tag in first_value or
                    quad_start_tag in first_value or
                    quad_end_tag in first_value
            ):
                print(f"Skipping entry due to first value: {first_value}")
                continue

        for convo in conversations:
            if first_one:
                first_one=False
                continue

            value = convo.get('value', '')
            if image_start_tag in value or image_end_tag in value or image_pad_tag in value or ref_end_tag in value or ref_start_tag in value  or box_end_tag in value or box_start_tag in value or quad_start_tag in value or quad_end_tag in value:
                # print("found bad ones: ", value)
                has_unmatched_img_tag = True
                break

        # Only include data points without unmatched <img> tags

        entry['conversations'] =json.dumps(conversations)

        if not has_unmatched_img_tag:
            filtered_data.append(entry)

    # Sort filtered data by total length of 'value' in conversations
    filtered_data.sort(
        key=lambda entry: len(json.dumps(entry, ensure_ascii=False)),
        reverse=True
    )

    # Save the filtered and sorted data to a new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(filtered_data, output_file, ensure_ascii=False, indent=4)

    print(f"final_length: {len(filtered_data)}")

    print(f"Filtered and sorted data saved to {output_file_path}")

# Replace 'sft_train.json' with the path to your input file and 'filtered_data.json' with the desired output file path
filter_out_bad_datapoints('./data/sft_train.json', './data/filtered_data.json')
