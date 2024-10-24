"""
This script evaluates generated text against reference summaries using a range of metrics, including BERTScore, semantic similarity, Word Mover's Distance (WMD), and BLEURT. The script is optimized for parallel execution and supports Arabic-language evaluation models.

Key Features:
- Loads generated summaries and reference data for comparison.
- Computes text metrics such as BERTScore, semantic similarity, WMD, and BLEURT in parallel.
- Logs the evaluation process and saves results in CSV format.
"""

import os
import json
import jsonlines
import logging
import bert_score
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import KeyedVectors
from multiprocessing import Pool
import torch
import argparse

# ============================
# 1. Setup and Configuration
# ============================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for generated summaries.")
    
    parser.add_argument("--eval_dir", type=str, default="eval_trl", help="Directory to save evaluation results.")
    parser.add_argument("--generated_data_path", type=str, required=True, help="Path to the generated outputs JSONL file.")
    parser.add_argument("--bertscore_model", type=str, default="bert-base-multilingual-cased", help="BERTScore model name.")
    parser.add_argument("--semantic_model", type=str, default="asafaya/bert-base-arabic", help="Semantic similarity model name.")
    parser.add_argument("--word2vec_model_path", type=str, required=True, help="Path to the pre-trained Word2Vec model for WMD.")
    parser.add_argument("--bleurt_checkpoint_path", type=str, required=True, help="Path to BLEURT checkpoint for BLEURT score computation.")
    parser.add_argument("--num_processes", type=int, default=16, help="Number of processes for parallel execution.")
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA (GPU) if specified.")
    
    return parser.parse_args()

def setup_environment(args):
    if args.disable_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Create the evaluation directory if it doesn't exist
    os.makedirs(args.eval_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.eval_dir, "evaluation_2.log"),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Starting evaluation process.")

# Initialize the global models for each parallel worker
def init_worker(bertscore_model_name, semantic_model_name, word2vec_model_path, bleurt_checkpoint_path):
    global bert_model
    global semantic_model_tuple
    global arabic_model
    global bleurt_scorer

    # BERTScore model
    bert_model = bertscore_model_name  # BERTScore works with model names

    # Semantic Similarity model
    semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
    semantic_model = AutoModel.from_pretrained(semantic_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic_model.to(device)
    semantic_model_tuple = (semantic_model, semantic_tokenizer)

    # Load Arabic word2vec model for WMD computation
    arabic_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

    # Load BLEURT model
    from bleurt import score
    bleurt_scorer = score.BleurtScorer(bleurt_checkpoint_path)

# ============================
# 2. Load Generated Data
# ============================

def load_generated_data(generated_data_path: str):
    """
    Loads generated data from a JSONL file.
    
    Args:
        generated_data_path (str): Path to the generated outputs JSONL file.
    
    Returns:
        list: A list of dictionaries representing the generated data.
    """
    try:
        with jsonlines.open(generated_data_path) as reader:
            generated_data = [obj for obj in reader]
        logging.info(f"Loaded {len(generated_data)} entries from '{generated_data_path}'.")
        return generated_data
    except Exception as e:
        logging.error(f"Failed to load generated data: {e}")
        raise e

# ============================
# 3. Define Metric Functions
# ============================

def extract_json_fields(json_text):
    """
    Extracts fields from a JSON string. If input is already a dictionary, it's returned as is.

    Args:
        json_text (str or dict): The input JSON text or dictionary.
    
    Returns:
        dict: Extracted JSON fields as a dictionary.
    """
    if isinstance(json_text, dict):
        return json_text
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logging.warning(f"JSON decoding failed: {e}")
        return {}
    except Exception as e:
        logging.warning(f"Unexpected error in JSON extraction: {e}")
        return {}

def compute_bert_score(candidate_texts: list, reference_texts: list, lang: str, model_type: str):
    """
    Computes BERTScore between candidate and reference texts.
    
    Args:
        candidate_texts (list): List of candidate texts.
        reference_texts (list): List of reference texts.
        lang (str): Language for BERTScore.
        model_type (str): Model type for BERTScore.
    
    Returns:
        list: List of F1 scores for BERTScore.
    """
    P, R, F1 = bert_score.score(candidate_texts, reference_texts, lang=lang, model_type=model_type, num_layers=12, verbose=False)
    return F1.tolist()

def compute_semantic_similarity(semantic_model_tuple, candidate_texts: list, reference_texts: list):
    """
    Computes semantic similarity between candidate and reference texts using a BERT-based model.
    
    Args:
        semantic_model_tuple (tuple): The semantic model and tokenizer.
        candidate_texts (list): List of candidate texts.
        reference_texts (list): List of reference texts.
    
    Returns:
        list: List of cosine similarities between embeddings of the candidate and reference texts.
    """
    model, tokenizer = semantic_model_tuple
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    candidate_embeddings = [get_embedding(text) for text in candidate_texts]
    reference_embeddings = [get_embedding(text) for text in reference_texts]

    similarities = [cosine_similarity([cand_emb], [ref_emb])[0][0] for cand_emb, ref_emb in zip(candidate_embeddings, reference_embeddings)]
    return similarities

def compute_wmd_similarity(reference, candidate, model):
    """
    Computes Word Mover's Distance (WMD) similarity between reference and candidate texts.
    
    Args:
        reference (str): The reference text.
        candidate (str): The candidate text.
        model (gensim.models.KeyedVectors): Pre-trained Word2Vec model.
    
    Returns:
        float: WMD similarity score.
    """
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    try:
        distance = model.wmdistance(reference_tokens, candidate_tokens)
        similarity = 1 / (1 + distance)  # Convert distance to similarity
        return similarity
    except Exception as e:
        logging.error(f"Error computing WMD: {e}")
        return 0

def compute_bleurt_score(candidate_text: str, reference_text: str):
    """
    Computes BLEURT score between candidate and reference texts.
    
    Args:
        candidate_text (str): The candidate text.
        reference_text (str): The reference text.
    
    Returns:
        float: BLEURT score.
    """
    global bleurt_scorer
    try:
        scores = bleurt_scorer.score(references=[reference_text], candidates=[candidate_text])
        return scores[0]
    except Exception as e:
        logging.error(f"Error computing BLEURT Score: {e}")
        return None

def get_nested_value(dictionary, keys):
    """
    Retrieves a nested value from a dictionary given a list of keys.

    Args:
        dictionary (dict): Input dictionary.
        keys (list): List of keys to retrieve nested value.

    Returns:
        Any: The nested value or None if keys don't exist.
    """
    for key in keys:
        if isinstance(dictionary, dict) and key in dictionary:
            dictionary = dictionary[key]
        else:
            return None
    return dictionary

def is_valid_json(json_string):
    """
    Checks if a string is valid JSON.

    Args:
        json_string (str): JSON string to check.
    
    Returns:
        bool: True if valid JSON, False otherwise.
    """
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False

# ============================
# 4. Compute Metrics in Parallel
# ============================

def compute_text_metrics(args):
    """
    Computes text-based metrics (BERTScore, semantic similarity, WMD, BLEURT) for a given pair of reference and candidate texts.
    
    Args:
        args (tuple): Tuple containing reference_text, candidate_text, and key.
    
    Returns:
        tuple: (key, metrics_dict) containing the evaluated metrics for the given key.
    """
    reference_text, candidate_text, key = args
    metrics_dict = {
        "bertscore": None,
        "semantic_similarity": None,
        "wmd_similarity": None,
        "bleurt_score": None
    }

    if reference_text is None or candidate_text is None:
        logging.warning(f"Skipping metrics computation due to None value. Reference: {reference_text}, Candidate: {candidate_text}")
        return key, metrics_dict

    reference_text = str(reference_text)
    candidate_text = str(candidate_text)

    if not reference_text.strip() or not candidate_text.strip():
        logging.warning(f"Skipping metrics computation due to empty string. Reference: '{reference_text}', Candidate: '{candidate_text}'")
        return key, metrics_dict

    # Access global models
    global bert_model
    global semantic_model_tuple
    global arabic_model

    # Compute BERTScore
    try:
        bert_score = compute_bert_score([candidate_text], [reference_text], lang='ar', model_type=bert_model)[0]
        metrics_dict["bertscore"] = bert_score
    except Exception as e:
        logging.error(f"Error computing BERTScore: {e}")

    # Compute Semantic Similarity
    try:
        sem_sim = compute_semantic_similarity(semantic_model_tuple, [candidate_text], [reference_text])[0]
        metrics_dict["semantic_similarity"] = sem_sim
    except Exception as e:
        logging.error(f"Error computing Semantic Similarity: {e}")

    # Compute WMD Similarity
    try:
        wmd_sim = compute_wmd_similarity(reference_text, candidate_text, arabic_model)
        metrics_dict["wmd_similarity"] = wmd_sim
    except Exception as e:
        logging.error(f"Error computing WMD Similarity: {e}")

    # Compute BLEURT Score
    try:
        bleurt_score = compute_bleurt_score(candidate_text, reference_text)
        metrics_dict["bleurt_score"] = bleurt_score
    except Exception as e:
        logging.error(f"Error computing BLEURT Score: {e}")

    return key, metrics_dict

def evaluate_generated_outputs(generated_data, bertscore_model_name, semantic_model_name, word2vec_model_path, bleurt_checkpoint_path, keys_to_evaluate):
    """
    Evaluates generated outputs against reference data using various metrics (BERTScore, semantic similarity, WMD, BLEURT).
    
    Args:
        generated_data (list): List of generated data containing prompts, reference summaries, and generated texts.
        bertscore_model_name (str): Model name for BERTScore evaluation.
        semantic_model_name (str): Model name for semantic similarity evaluation.
        word2vec_model_path (str): Path to pre-trained Word2Vec model for WMD computation.
        bleurt_checkpoint_path (str): Path to BLEURT checkpoint for BLEURT score computation.
        keys_to_evaluate (list): List of keys to evaluate.
    
    Returns:
        dict: Dictionary containing the computed metrics.
    """
    metrics = {key: {
        "bertscore": [],
        "semantic_similarity": [],
        "wmd_similarity": [],
        "bleurt_score": []
    } for key in keys_to_evaluate}
    metrics["json_validity"] = []

    total_samples = len(generated_data)
    args_list = []

    for idx, data in enumerate(generated_data):
        try:
            prompt = data["prompt"]
            reference_summary = data["reference_summary"]
            generated_text = data["generated_text"]

            # Deserialize JSON
            reference_json = json.loads(reference_summary)
            generated_json_text = generated_text

            # Extract generated JSON
            if "```json" in generated_json_text:
                generated_json_text = generated_json_text.split("```json")[1].split("```")[0].strip()
            else:
                start_idx = generated_json_text.find('{')
                end_idx = generated_json_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    generated_json_text = generated_json_text[start_idx:end_idx+1]
                else:
                    raise ValueError("No JSON detected in generated text.")

            # Check JSON validity
            json_validity = is_valid_json(generated_json_text)
            metrics["json_validity"].append(int(json_validity))

            generated_json = extract_json_fields(generated_json_text)

            # Evaluate the keys
            for key in keys_to_evaluate:
                reference_value = None
                candidate_value = None

                if key in ['court', 'main_case_topic']:
                    reference_value = get_nested_value(reference_json, ['case_information', key])
                    candidate_value = get_nested_value(generated_json, ['case_information', key])
                elif key == 'persons_involved':
                    reference_value = get_nested_value(reference_json, ['persons_involved'])
                    candidate_value = get_nested_value(generated_json, ['persons_involved'])
                elif key in ['overview', 'relevant_dates']:
                    reference_value = get_nested_value(reference_json, ['background_of_the_case', key])
                    candidate_value = get_nested_value(generated_json, ['background_of_the_case', key])
                elif key == 'key_issues':
                    reference_value = get_nested_value(reference_json, ['key_issues'])
                    candidate_value = get_nested_value(generated_json, ['key_issues'])
                elif key in ['claimants_arguments', 'defendants_arguments']:
                    reference_value = get_nested_value(reference_json, ['arguments_presented', key])
                    candidate_value = get_nested_value(generated_json, ['arguments_presented', key])
                elif key in ['evidence_reviewed', 'rulings_made', 'legal_principles_applied']:
                    reference_value = get_nested_value(reference_json, ['courts_findings', key])
                    candidate_value = get_nested_value(generated_json, ['courts_findings', key])
                elif key in ['final_decision', 'implications']:
                    reference_value = get_nested_value(reference_json, ['outcome', key])
                    candidate_value = get_nested_value(generated_json, ['outcome', key])

                if reference_value is not None and candidate_value is not None:
                    if isinstance(reference_value, list):
                        reference_value = ' '.join(map(str, reference_value))
                    if isinstance(candidate_value, list):
                        candidate_value = ' '.join(map(str, candidate_value))

                    args_list.append((reference_value, candidate_value, key))
                else:
                    logging.warning(f"Skipping metrics computation for key '{key}' due to None value.")
        except Exception as e:
            logging.error(f"Error processing data at index {idx}: {e}")
            continue

    # Compute metrics in parallel
    num_processes = 16
    with Pool(processes=num_processes, initializer=init_worker, initargs=(bertscore_model_name, semantic_model_name, word2vec_model_path, bleurt_checkpoint_path)) as pool:
        results = list(tqdm(pool.imap(compute_text_metrics, args_list), total=len(args_list), desc="Computing Metrics"))

    # Assign results to metrics
    for key in keys_to_evaluate:
        metrics[key]["bertscore"] = []
        metrics[key]["semantic_similarity"] = []
        metrics[key]["wmd_similarity"] = []
        metrics[key]["bleurt_score"] = []

    for key_result, metrics_dict in results:
        metrics[key_result]["bertscore"].append(metrics_dict["bertscore"])
        metrics[key_result]["semantic_similarity"].append(metrics_dict["semantic_similarity"])
        metrics[key_result]["wmd_similarity"].append(metrics_dict["wmd_similarity"])
        metrics[key_result]["bleurt_score"].append(metrics_dict["bleurt_score"])

    return metrics

# ============================
# 5. Aggregate and Analyze Metrics
# ============================

def safe_mean(score_list):
    """
    Safely computes the mean of a list, ignoring None values.

    Args:
        score_list (list): List of scores.

    Returns:
        float: The mean of the valid scores or None if no valid scores.
    """
    valid_scores = [s for s in score_list if s is not None]
    if len(valid_scores) == 0:
        return None
    else:
        return np.mean(valid_scores)

def aggregate_metrics(metrics: dict):
    """
    Aggregates the computed metrics into a dataframe for analysis.
    
    Args:
        metrics (dict): Dictionary containing the computed metrics.
    
    Returns:
        pd.DataFrame: Dataframe containing the aggregated metrics.
    """
    metrics_list = []
    for key, scores in metrics.items():
        if key == "json_validity":
            score = safe_mean(scores)
            metric_dict = {
                "Key": "JSON Validity",
                "Score": round(score, 2) if score is not None else None  # Convert to percentage
            }
        else:
            bertscore_mean = safe_mean(scores["bertscore"])
            semantic_similarity_mean = safe_mean(scores["semantic_similarity"])
            wmd_similarity_mean = safe_mean(scores["wmd_similarity"])
            bleurt_mean = safe_mean(scores["bleurt_score"])

            metric_dict = {
                "Key": key,
                "BERTScore F1": round(bertscore_mean, 4) if bertscore_mean is not None else None,
                "Semantic Similarity": round(semantic_similarity_mean, 4) if semantic_similarity_mean is not None else None,
                "WMD Similarity": round(wmd_similarity_mean, 4) if wmd_similarity_mean is not None else None,
                "BLEURT Score": round(bleurt_mean, 4) if bleurt_mean is not None else None
            }
        metrics_list.append(metric_dict)

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df

# ============================
# 6. Main Execution Function
# ============================

def main():
    """
    Main function to execute the evaluation process.
    """
    args = parse_args()
    setup_environment(args)

    generated_data = load_generated_data(args.generated_data_path)

    keys_to_evaluate = [
        'court', 'main_case_topic',
        'overview', 'relevant_dates', 'key_issues',
        'claimants_arguments', 'defendants_arguments', 'evidence_reviewed',
        'rulings_made', 'legal_principles_applied', 'final_decision',
        'implications'
    ]

    metrics = evaluate_generated_outputs(
        generated_data=generated_data,
        bertscore_model_name=args.bertscore_model,
        semantic_model_name=args.semantic_model,
        word2vec_model_path=args.word2vec_model_path,
        bleurt_checkpoint_path=args.bleurt_checkpoint_path,
        keys_to_evaluate=keys_to_evaluate
    )

    metrics_df = aggregate_metrics(metrics)
    print("\n=== Evaluation Metrics ===\n")
    print(metrics_df.to_string(index=False))

    # Save the evaluation metrics to a CSV file
    metrics_csv_path = os.path.join(args.eval_dir, "evaluation_metrics.csv")
    try:
        metrics_df.to_csv(metrics_csv_path, index=False)
        logging.info(f"Saved evaluation metrics to '{metrics_csv_path}'.")
    except Exception as e:
        logging.error(f"Failed to save evaluation metrics: {e}")

    logging.info("Evaluation process completed.")

if __name__ == "__main__":
    main()