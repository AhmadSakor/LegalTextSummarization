# Legal Case Analysis and Summarization System 🔍

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive system for collecting, analyzing, and summarizing legal cases from Morocco's Judicial Portal using Large Language Models fine tuning and knowledge graph techniques.

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage Pipeline](#usage-pipeline)
- [Project Structure](#project-structure)
- [Performance](#performance)

## 🔍 Overview

This project provides an end-to-end solution for processing legal cases, from data collection to serving structured summaries via an API. The system uses state-of-the-art language models, specifically a fine-tuned LLaMA model, to generate comprehensive case summaries and builds a knowledge graph for advanced legal analysis.

## 🤗 Hugging Face Model

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Llama3.2--3B--Instruct--Legal--Summarization-yellow)](https://huggingface.co/ahmadsakor/Llama3.2-3B-Instruct-Legal-Summarization)

The fine-tuned model is publicly available on the Hugging Face Hub: [ahmadsakor/Llama3.2-3B-Instruct-Legal-Summarization](https://huggingface.co/ahmadsakor/Llama3.2-3B-Instruct-Legal-Summarization)


## 🏗️ System Architecture

The system consists of five main components:

1. **Data Collection & Preparation**: Automated crawling and OCR processing of legal documents
2. **Model Fine-tuning**: Custom LLaMA model adaptation for legal summarization
3. **Evaluation Framework**: Comprehensive metrics for model assessment
4. **Knowledge Graph**: RDF-based graph database for legal case analysis
5. **Inference API**: FastAPI service for generating case summaries

## ✨ Key Features

- Automated legal case collection from the Moroccan Judicial Portal
- Advanced OCR processing with error correction
- Fine-tuned LLaMA model for Arabic legal text
- Comprehensive evaluation framework
- RDF-based knowledge graph
- FastAPI-based inference service
- Multi-language support (Arabic/English)

## 💻 Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (12GB+ VRAM recommended)
- 32GB+ RAM recommended
- 50GB+ storage space
- Internet connection
- GraphDB instance (for knowledge graph)

## 🚀 Installation

Each component has its own dependencies. To set up the complete system:

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Follow the instructions in each directory.


## 📝 Usage Pipeline


1. Data Collection & Preparation


2. Model Fine-tuning


3. Model Evaluation


4. Knowledge Graph Creation


5. Deploy Inference API


## 📁 Project Structure

```
├── fetch_data/          # Data collection and processing
├── fine_tuning/         # Model training and adaptation
├── evaluation/          # Performance assessment
├── knowledge_graph/     # Graph database creation
├── inference/           # API service
├── data/               # Data storage
└── logs/               # System logs
```

## 📈 Performance

The fine-tuned model shows significant improvements over the base model:

- Court Information: +4.58% BERTScore F1
- Legal Principles: +10.48% BERTScore F1
- Final Decision: +9.27% BERTScore F1
- JSON Validity: 99% accuracy

Full evaluation metrics available in the evaluation directory.

## 🙏 Acknowledgments

- [Moroccan Judicial Portal](https://juriscassation.cspj.ma/en) for providing access to legal cases
- Contributors and maintainers of the OCR libraries used in this project
- Contributors and maintainers of 🤗 Transformers
- Contributors and maintainers of DeepSpeed
- Contributors and maintainers of PEFT (Parameter-Efficient Fine-Tuning)
- Contributors and maintainers of Weights & Biases for experiment tracking
- Contributors and maintainers of Moroccan Judicial Portal for providing access to legal cases
- Contributors and maintainers of Meta AI for the base LLaMA model.