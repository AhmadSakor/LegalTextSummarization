#  Model Evaluation 📊

A comprehensive evaluation framework for assessing the performance of the base and fine-tuned LLaMA models on legal case summarization tasks.

## 📋 Table of Contents
- [Overview](#-overview)
- [Installation](#-installation)
- [Evaluation Process](#-evaluation-process)
- [Running Evaluations](#-running-evaluations)
- [Evaluation Metrics](#-evaluation-metrics)
- [Performance Results](#-performance-results)
- [Key Improvements](#-key-improvements)

## 🔍 Overview

This evaluation framework provides a detailed assessment of model performance using multiple complementary metrics, specifically designed for Arabic legal text summarization. The evaluation demonstrates significant improvements in the fine-tuned model across various aspects of legal text understanding and generation.


## 🚀 Installation

1. We need to download fasttext word vector model for the Arabic language from [here](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.bin.gz)

2. After unzipping the downloaded model we need to convert it to gensim compatible model using the following script
```bash
python helpers/fasttext2gensim.py
```

## 📈 Evaluation Process

### 1. Generate Model Outputs

For Base Model:
```bash
python generate_output.py \
    --eval_dir ../data/evaluation/base_model/ \
    --model_id meta-llama/Llama-3.2-3B-Instruct \
    --eval_dataset_path ../data/eval_dataset.jsonl \
    --batch_size 16
```

For Fine-tuned Model:
```bash
python generate_output.py \
    --eval_dir ../data/evaluation/fine_tuned_model/ \
    --model_id ./fine-tuned-llama \
    --eval_dataset_path ../data/eval_dataset.jsonl \
    --batch_size 16
```

### 2. Compute Metrics

For both models:
```bash
python compute_metrics.py \
    --eval_dir ../data/evaluation/<model_dir>/ \
    --generated_data_path ../data/evaluation/<model_dir>/generated_outputs.jsonl \
    --bertscore_model bert-base-multilingual-cased \
    --semantic_model asafaya/bert-base-arabic \
    --word2vec_model_path helpers/arabic_word2vec_model.bin \
    --bleurt_checkpoint_path /helpers/bleurt_checkpoints/BLEURT-20 \
    --num_processes 16 \
    --disable_cuda  # Optional
```

## 📊 Evaluation Metrics

### BERTScore F1
- Evaluates semantic similarity using contextual embeddings
- Particularly suitable for Arabic text evaluation
- Captures precise meaning in legal context
- Uses multilingual BERT model for computation

### Semantic Similarity
- Measures overall semantic closeness between generated and reference summaries
- Essential for ensuring preserved meaning in legal summaries
- Uses Arabic BERT model for accurate Arabic text understanding

### WMD (Word Mover's Distance) Similarity
- Calculates minimum distance between text embeddings
- Handles different phrasings of same legal concepts
- More forgiving for semantically similar but differently worded phrases
- Based on Arabic Word2Vec embeddings

### BLEURT Score
- Learned metric correlating with human judgments
- Captures nuanced differences in meaning
- Assesses fluency, coherence, and factual consistency
- Uses BLEURT-20 checkpoint

### JSON Validity
- Measures structural correctness of generated output
- Critical for downstream processing
- Ensures maintained structured format

## 📈 Performance Results

### Base Model Performance

| Key | BERTScore F1 | Semantic Similarity | WMD Similarity | BLEURT Score | Score |
|-----|-------------|-------------------|---------------|--------------|--------|
| Court | 0.9056 | 0.9399 | 0.8555 | 0.8281 | - |
| Main Case Topic | 0.6631 | 0.8037 | 0.5788 | 0.3977 | - |
| Legal Principles Applied | 0.7133 | 0.8350 | 0.6316 | 0.4954 | - |
| Final Decision | 0.6221 | 0.7654 | 0.5346 | 0.3464 | - |
| JSON Validity | - | - | - | - | 0.95 |

### Fine-tuned Model Performance

| Key | BERTScore F1 | Semantic Similarity | WMD Similarity | BLEURT Score | Score |
|-----|-------------|-------------------|---------------|--------------|--------|
| Court | **0.9514** | **0.9709** | **0.9265** | **0.9165** | - |
| Main Case Topic | **0.7616** | **0.8612** | **0.6733** | **0.5664** | - |
| Legal Principles Applied | **0.8181** | **0.9046** | **0.7496** | **0.6648** | - |
| Final Decision | **0.7148** | **0.8624** | **0.5889** | **0.5424** | - |
| JSON Validity | - | - | - | - | **0.99** |

## 🎯 Key Improvements

The fine-tuned model showed significant improvements across all metrics:

1. **Court Information**: 
   - BERTScore F1: +4.58%
   - Semantic Similarity: +3.10%
   - WMD Similarity: +7.10%
   - BLEURT Score: +8.84%

2. **Legal Principles**:
   - BERTScore F1: +10.48%
   - Semantic Similarity: +6.96%
   - WMD Similarity: +11.80%
   - BLEURT Score: +16.94%

3. **Final Decision**:
   - BERTScore F1: +9.27%
   - Semantic Similarity: +9.70%
   - WMD Similarity: +5.43%
   - BLEURT Score: +19.60%

4. **Overall Improvements**:
- Implications accuracy: +26.16%
- Case overview quality: +10.75%
- Arguments comprehension: +9.00%
- JSON validity: +4.00%

## 📋 Evaluation Requirements

### Models Required
- `bert-base-multilingual-cased`: For BERTScore computation
- `asafaya/bert-base-arabic`: For semantic similarity
- Arabic Word2Vec model: For WMD calculation
- BLEURT-20 checkpoint: For BLEURT score computation

### Hardware Requirements
- 16GB+ RAM recommended
- Multi-core CPU for parallel processing
- GPU optional (can be disabled with `--disable_cuda`)

## 📊 Significance

The evaluation results demonstrate:
1. Consistent improvements across all metrics
2. Strong performance in critical legal content understanding
3. Enhanced structural consistency in output
4. Significant gains in semantic accuracy
5. Improved handling of Arabic legal terminology

