# Serving the model using an API 🚀

A FastAPI service for generating structured summaries of legal cases using our fine-tuned LLaMA model. The API provides a single endpoint that validates legal text input and generates comprehensive JSON-formatted summaries.

## 📋 Table of Contents
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Model Details](#model-details)
- [Troubleshooting](#troubleshooting)

## 💻 System Requirements

### Hardware Requirements
- **CPU**: 4+ cores recommended
- **RAM**: Minimum 16GB, 32GB recommended
- **GPU**: 
  - NVIDIA GPU with 12GB+ VRAM recommended
  - CUDA compatibility required for GPU inference
  - CPU-only inference possible but slower
- **Storage**: Minimum 10GB free space

### Software Requirements
- Python 3.10+
- CUDA Toolkit 11.7+ (for GPU support)
- Ubuntu 20.04+ or similar Linux distribution

## 🚀 Installation

1. Install system dependencies:
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev nginx gunicorn
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```



## 🔧 Running the API

### Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
gunicorn -w 1 -k uvicorn.workers.UvicornWorker \
    --timeout 500 \
    --bind 0.0.0.0:8000 \
    main:app
```

Configuration explained:
- `-w 1`: Single worker due to model memory requirements
- `--timeout 500`: Extended timeout for long text processing
- `--bind 0.0.0.0:8000`: Listen on all interfaces, port 8000

## 📚 API Documentation

### Endpoint: `/predict/`
- **Method**: POST
- **Content-Type**: application/json
- **Request Body**:
```json
{
    "input": "Your legal text here..."
}
```

### Response Format
```json
{
    "generated_text": "Generated JSON summary..."
}
```

## 🔍 Usage Examples

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict/" \
     -H "Content-Type: application/json" \
     -d '{
           "input": "قرار عدد: 305/3 في الملف عدد: 91/3/1/2023 بتاريخ: 04/07/2023..."
         }'
```

### Using Python Requests
```python
import requests

url = "http://localhost:8000/predict/"
payload = {
    "input": "قرار عدد: 305/3 في الملف عدد: 91/3/1/2023 بتاريخ: 04/07/2023..."
}
response = requests.post(url, json=payload)
print(response.json())
```

## 🤖 Model Details

The API uses:
- Model: `ahmadsakor/Llama3.2-3B-Instruct-Legal-Summarization`
- Architecture: LLaMA 3.2 3B parameters
- Features:
  - Automatic GPU/CPU detection
  - BFloat16/Float32 precision based on hardware
  - Left-padding for efficient batch processing
  - Maximum 1500 output tokens


## ⚠️ Troubleshooting

Common issues and solutions:

1. **Memory Errors**
   ```bash
   # Reduce model memory usage
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
   ```

2. **CUDA Out of Memory**
   - Ensure no other processes are using GPU
   - Try running on CPU if GPU memory is insufficient

3. **Slow Response Times**
   - Check GPU utilization
   - Monitor system resources
   - Consider reducing max_new_tokens

## 🔒 Security Notes

- API has no built-in authentication
- Consider adding authentication for production
- Implement rate limiting for public deployments
- Validate input size limits

