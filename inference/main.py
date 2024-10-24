"""
FastAPI service for summarizing legal texts using a fine-tuned Llama model. The API uses a text generation pipeline to generate structured summaries of legal cases in JSON format. 

Key Features:
- Validates whether the input text is related to a legal case.
- Summarizes valid legal texts according to a predefined template.
- Leverages the Hugging Face Transformers library for efficient model loading and inference.
"""

from fastapi import FastAPI, Request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Model ID for the fine-tuned legal summarization model
model_id = "ahmadsakor/Llama3.2-3B-Instruct-Legal-Summarization"

# Set the device map based on GPU availability
device_map = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
tokenizer.padding_side = 'left'  # Left padding for batch alignment

# Load the model with the appropriate dtype and device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device_map
)

# Create the text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=dtype,
    device_map=device_map
)

# System prompt for the AI assistant's task
system_prompt = """
You are a legal assistant AI that summarizes legal cases in JSON format following a specific template. 
Please ensure all outputs are structured and all keys are in English while the values are in Arabic. 
Be concise, informative, and follow the template strictly.
"""

# Template prompt for the legal text summary
template_prompt="""
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
Example of output json format:
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
  "key_issues": [
  ],
  "arguments_presented": {
    "claimants_arguments": "",
    "defendants_arguments": ""
  },
  "courts_findings": {
    "evidence_reviewed": "",
    "rulings_made": "",
    "legal_principles_applied": [
    ]
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
Input:\n
"""


def validate_legal_text(text: str) -> bool:
    """
    Validates whether the input text is related to a legal case by using a brief validation pipeline.
    The pipeline only returns a 'yes' or 'no' response.
    
    Args:
        text (str): The input text to be validated.
    
    Returns:
        bool: True if the text is legal, False otherwise.
    """
    validation_prompt = (
        "You are an assistant that can only say two words (yes or no). You choose yes if the input is a text related to a legal case. Otherwise, you say no.\n"
        "Please answer yes or no if the following input is a text related to a legal case. "
        "If you are unable to decide answer no.\n\n"
        f"Input: {text}\nAnswer (yes or no): "
    )

    messages = [{"role": "user", "content": validation_prompt}]
    outputs = pipe(messages, max_new_tokens=10)
    
    # Extract the yes/no response and normalize it
    response = outputs[0]["generated_text"].strip().lower()
    return "yes" in response  # Return True if the response is "yes"

@app.post("/predict/")
async def predict(request: Request):
    """
    Endpoint to predict and summarize the legal text input. Validates whether the text is legal and, if so, 
    generates a structured summary using the model.
    
    Args:
        request (Request): The incoming HTTP request containing the input text.
    
    Returns:
        dict: A JSON response containing the generated summary or an error message if the input is invalid.
    """
    request_data = await request.json()
    inputs = request_data.get("input", "")

    # Validate if the input text is a legal case
    if not validate_legal_text(inputs):
        return {"generated_text": "Sorry, I can only assist you with legal text summarization."}

    # If valid, proceed with generating the summary
    full_text = inputs
    user_part = template_prompt + '\n' + full_text
    prompt = (
        f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}\n"
        f"<|start_header_id|>user<|end_header_id|>\n{user_part}\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    # Generate the summary using the pipeline
    generated_outputs = pipe(
        prompt,
        max_new_tokens=1500,  # Limit the number of tokens in the output
        num_return_sequences=1,  # Return a single sequence
        pad_token_id=pipe.tokenizer.eos_token_id,
        padding=True,
        return_full_text=False,
    )

    return {"generated_text": generated_outputs[0]["generated_text"]}
