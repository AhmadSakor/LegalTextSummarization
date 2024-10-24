import os
import json
import re
import argparse

def fix_full_text(json_string):
    """
    Corrects issues in the 'full_text' field of a JSON string by:
    - Replacing escaped double quotes with regular double quotes inside the 'full_text' value.
    - Converting unescaped double quotes to single quotes to maintain JSON structure integrity.

    Args:
        json_string (str): The input JSON string that may have incorrect or malformed 'full_text' field.

    Returns:
        str: The modified JSON string with corrected 'full_text' content.
    """
    # Regex replacement function to fix quote issues in the 'full_text' field
    def replace_quotes_in_full_text(match):
        full_text = match.group(1)
        # Replace escaped double quotes with regular double quotes, then convert other quotes to single quotes
        fixed_full_text = full_text.replace('\\"', '"').replace('"', "'")
        return f'"full_text": "{fixed_full_text}"'

    # Apply regex to capture and replace 'full_text' contents, handling special characters, quotes, and newlines
    fixed_json_string = re.sub(r'"full_text":\s*"(.*?)"\s*(?=,|})', replace_quotes_in_full_text, json_string, flags=re.DOTALL)
    
    return fixed_json_string

def process_files(input_folder, output_folder):
    """
    Processes all files in the input folder, fixes the JSON format in each, and saves the corrected content to the output folder.

    Args:
        input_folder (str): Path to the folder containing .txt files with erroneous JSON strings.
        output_folder (str): Path to the folder where fixed JSON files will be saved.
    
    The function reads each file, extracts the JSON content, applies the fix, and saves the corrected JSON.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):  # Process only .txt files (assuming error files are in .txt format)
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")

            with open(input_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Locate the JSON part after "Original GPT Response:"
            json_start = content.find("Original GPT Response:") + len("Original GPT Response:")
            json_string = content[json_start:].strip()

            # Attempt to fix and parse the JSON string
            try:
                fixed_json_string = fix_full_text(json_string)
                parsed_json = json.loads(fixed_json_string)

                # Save the corrected JSON content
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    json.dump(parsed_json, outfile, ensure_ascii=False, indent=2)

                print(f"Successfully processed and saved: {output_path}")
            except json.JSONDecodeError as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process JSON error files and fix full_text field issues.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing erroneous JSON files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder where fixed JSON files will be saved.")
    
    # Parse arguments
    args = parser.parse_args()

    # Execute the process
    process_files(args.input_folder, args.output_folder)
