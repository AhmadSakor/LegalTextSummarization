import os
import json
import asyncio
import aiohttp
import aiofiles
from datetime import datetime
from time import time
import argparse

class RateLimiter:
    """
    Implements a token bucket algorithm to rate limit API calls.

    Args:
        rate_limit (int): Maximum number of requests allowed per period.
        period (int): The time window for the rate limit in seconds.
    """
    
    def __init__(self, rate_limit, period):
        self.rate_limit = rate_limit
        self.period = period
        self.tokens = rate_limit
        self.last_refill = time()

    async def acquire(self):
        """
        Acquires a token if available, otherwise waits for enough tokens to refill.
        """
        while True:
            now = time()
            time_passed = now - self.last_refill

            # Refill tokens based on time passed and rate limit
            self.tokens += time_passed * (self.rate_limit / self.period)
            if self.tokens > self.rate_limit:
                self.tokens = self.rate_limit
            self.last_refill = now

            # Check if at least one token is available
            if self.tokens >= 1:
                self.tokens -= 1
                return
            else:
                wait_time = (1 - self.tokens) / (self.rate_limit / self.period)
                await asyncio.sleep(wait_time)


async def call_openai_api(session, api_key, system_prompt, user_prompt, rate_limiter):
    """
    Sends a request to OpenAI's GPT-4 API to generate a completion based on prompts.
    """
    await rate_limiter.acquire()  # Ensure rate limit is respected before making the request

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    # Define the request payload
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    async with session.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data) as response:
        response.raise_for_status()
        result = await response.json()
        return result['choices'][0]['message']['content']


async def process_file(session, input_folder, output_folder, filename, system_prompt, template_prompt, error_log, api_key, rate_limiter, errors_folder_path):
    """
    Processes a single text file, sends the content to GPT-4, and saves the response in JSON format.
    """
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename.replace('.txt', '.json'))
    file_id = os.path.splitext(filename)[0]
    errors_folder = errors_folder_path
    os.makedirs(errors_folder, exist_ok=True)

    if os.path.exists(output_path) or os.path.exists(os.path.join(errors_folder, f"{file_id}.txt")):
        print(f"Skipping {filename} as it has already been processed.")
        return

    try:
        async with aiofiles.open(input_path, 'r', encoding='utf-8') as file:
            case_text = await file.read()

        user_prompt = case_text.strip('"').strip("'") + "\n\n" + template_prompt

        response_content = await call_openai_api(session, api_key, system_prompt, user_prompt, rate_limiter)
        response_content = response_content.strip('```json').strip('```').strip()

        response_json = json.loads(response_content)

        async with aiofiles.open(output_path, 'w', encoding='utf-8') as json_file:
            await json_file.write(json.dumps(response_json, ensure_ascii=False, indent=4))

        print(f"Processed and saved: {output_path}")

    except json.JSONDecodeError as json_error:
        error_file_path = os.path.join(errors_folder, f"{file_id}.txt")
        async with aiofiles.open(error_file_path, 'w', encoding='utf-8') as error_file:
            await error_file.write(f"JSON Decode Error:\n{str(json_error)}\n\nOriginal GPT Response:\n{response_content}")

        async with aiofiles.open(error_log, 'a', encoding='utf-8') as log_file:
            await log_file.write(f"{datetime.now()}: JSON decode error processing {filename}: {str(json_error)}\n")

    except Exception as e:
        error_file_path = os.path.join(errors_folder, f"{file_id}.txt")
        async with aiofiles.open(error_file_path, 'w', encoding='utf-8') as error_file:
            await error_file.write(f"Error:\n{str(e)}\n\nOriginal GPT Response:\n{response_content}")

        async with aiofiles.open(error_log, 'a', encoding='utf-8') as log_file:
            await log_file.write(f"{datetime.now()}: Error processing {filename}: {str(e)}\n")


async def process_case_files(input_folder, output_folder, error_log, api_key, max_requests_per_minute, rate_limit_period, errors_folder):
    """
    Main function to process all case files asynchronously.
    """
    os.makedirs(output_folder, exist_ok=True)

    rate_limiter = RateLimiter(max_requests_per_minute, rate_limit_period)

    system_prompt = (
        "I have this legal text by doing OCR over Arabic text. The text has some errors because of the OCR and some sentences don't make sense. "
        "The text is from Morocco and related to the legal domain. Your task is to fix the errors in the text by predicting what was likely the original text "
        "and then summarize the case using the given template."
    )

    template_prompt = """
1. Fix the errors in the text by trying to predict what was the original text...
"""

    async with aiohttp.ClientSession() as session:
        tasks = []
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                task = process_file(session, input_folder, output_folder, filename, system_prompt, template_prompt, error_log, api_key, rate_limiter,errors_folder)
                tasks.append(task)
        await asyncio.gather(*tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process legal case files using GPT-4 API.')

    parser.add_argument('--api_key', type=str, required=True, help='Your OpenAI API key.')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing input text files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the JSON output files.')
    parser.add_argument('--error_log', type=str, default='errors_processing.log', help='File to log errors during processing.')
    parser.add_argument('--errors_folder', type=str, required=True, help='Folder to save files with errors.')
    parser.add_argument('--max_requests_per_minute', type=int, default=90, help='Maximum number of API requests allowed per minute.')
    parser.add_argument('--rate_limit_period', type=int, default=60, help='Rate limit period in seconds.')

    args = parser.parse_args()

    asyncio.run(process_case_files(args.input_folder, args.output_folder, args.error_log, args.api_key, args.max_requests_per_minute, args.rate_limit_period, args.errors_folder))
