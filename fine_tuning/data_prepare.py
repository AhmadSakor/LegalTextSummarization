import os
import json
from tqdm import tqdm
from datasets import Dataset
import argparse

# Function to build the complete prompt
def prepare_example(system_prompt, template_prompt, full_text, expected_summary):
    """
    Builds a complete prompt that includes system prompt, user input (legal case text), and assistant output (summary).

    Args:
        system_prompt (str): The prompt that frames the task for the AI.
        template_prompt (str): The structured template prompt for summarizing the legal case.
        full_text (str): The full legal text of the case.
        expected_summary (str): The expected output summary in JSON format.

    Returns:
        str: The full prompt for the example.
    """
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Summarize the following legal text in the JSON format:\n{template_prompt}\n{full_text}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{expected_summary}<|eot_id|>"
    )
    return prompt


def process_json_files(data_dir, output_file):
    # Define the system prompt that frames the task for the AI model
    system_prompt = (
        "You are a legal assistant AI that summarizes legal cases in JSON format following a specific template. "
        "Please ensure all outputs are structured and all keys are in English while the values are in Arabic. "
        "Be concise, informative, and follow the template strictly."
    )

    # Define the template prompt that outlines the JSON structure for summarizing legal cases
    template_prompt = """
    ###
    Legal Text Summary Template
    1. Case Information

    Case Number: [Insert case number]
    Date of Ruling: [Insert date of ruling]
    Court: [Insert court name]
    Main Case Topic: [Mention the main topic of the case]
    Parties Involved: [Insert names of parties]

    2. Persons involved including their:
     [List the Persons in the text including their roles in a structured format (Name, Role)]

    3. Background of the Case

    Overview: [Briefly describe the nature of the case and context]
    List of Relevant Dates with corresponding events in Arabic (Date, Event).

    4. Key Issues

    [List the main legal issues or disputes in the case]

    5. Arguments Presented

    Claimant’s Arguments:
    [Summarize the arguments made by the claimant]
    Defendant’s Arguments:
    [Summarize the arguments made by the defendant]

    6. Court's Findings

    Evidence Reviewed: [Mention the evidence the court relied on]
    Rulings Made: [Summarize the rulings made by the court]
    Legal Principles Applied: [List any relevant legal principles or statutes cited]

    7. Outcome

    Final Decision: [Describe the court's final decision]
    Implications: [Discuss any implications of the ruling]

    8. Additional Notes

    [Any additional observations or relevant information that should be noted]
    #####
    Example of output JSON format:
    {
      "case_information": {
        "case_number": "",
        "date_of_ruling": "",
        "court": "",
        "main_case_topic": "",
        "parties_involved": ""
      },
      "persons_involved": [
        {
          "name": "",
          "role": ""
        }
      ],
      "background_of_the_case": {
        "overview": "",
        "relevant_dates": [
          {
            "date": "",
            "event": ""
          }
        ]
      },
      "key_issues": [],
      "arguments_presented": {
        "claimants_arguments": "",
        "defendants_arguments": ""
      },
      "courts_findings": {
        "evidence_reviewed": "",
        "rulings_made": "",
        "legal_principles_applied": []
      },
      "outcome": {
        "final_decision": "",
        "implications": ""
      },
      "additional_notes": {
        "observations": ""
      }
    }
    ###
    """

    # Gather all JSON files from the specified directory
    json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]

    # Initialize a list to hold the dataset entries
    data = []

    # Process each JSON file
    for json_file in tqdm(json_files, desc='Processing JSON files'):
        with open(json_file, 'r', encoding='utf-8') as infile:
            data_content = json.load(infile)
            full_text = data_content.get('full_text', '')

            # Extract sections for building the expected summary
            case_information = data_content.get('case_information', {})
            persons_involved = data_content.get('persons_involved', [])
            background = data_content.get('background_of_the_case', {})
            key_issues = data_content.get('key_issues', [])
            arguments_presented = data_content.get('arguments_presented', {})
            courts_findings = data_content.get('courts_findings', {})
            outcome = data_content.get('outcome', {})
            additional_notes = data_content.get('additional_notes', {})

            # Prepare the expected summary in JSON format
            expected_summary = json.dumps({
                "case_information": case_information,
                "persons_involved": persons_involved,
                "background_of_the_case": background,
                "key_issues": key_issues,
                "arguments_presented": arguments_presented,
                "courts_findings": courts_findings,
                "outcome": outcome,
                "additional_notes": additional_notes,
            }, ensure_ascii=False, indent=2)

            # Build the complete prompt for this example
            example = prepare_example(system_prompt, template_prompt, full_text, expected_summary)

            # Add the prepared prompt to the dataset list
            data.append({"text": example})

    # Create the dataset in a format compatible with the TRL library
    my_dataset = Dataset.from_list(data)

    # Save the dataset to a JSONL file
    my_dataset.to_json(output_file, force_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Process legal case JSON files to prepare a fine-tuning dataset.")

    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the JSON files.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output JSONL file.")
    
    args = parser.parse_args()

    process_json_files(args.data_dir, args.output_file)


if __name__ == "__main__":
    main()
