# Langchain-Builder: FastAPI Service for LLM Applications

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/RahulPatnaik/Langchain-Builder)](https://github.com/RahulPatnaik/Langchain-Builder/stargazers)

## 🚀 Introduction

Langchain-Builder is a production-ready FastAPI service that provides a robust foundation for building applications powered by Large Language Models (LLMs) using LangChain. This template integrates multiple LLM providers, vector stores, caching, and observability features to accelerate your AI application development.

## ✨ Key Features

- **Multi-LLM Support**: OpenAI, Google Gemini, and Groq integration
- **RAG Pipeline**: Document processing and querying with ChromaDB
- **Asynchronous Architecture**: High-performance async/await implementation
- **Enterprise Ready**: Redis caching, S3 storage, and OpenTelemetry tracing
- **Security**: API key authentication and rate limiting
- **Streaming Support**: Real-time LLM response streaming

## 📦 Installation

### Prerequisites

- Python 3.9+
- Redis server
- LLM API keys (OpenAI/Gemini/Groq)

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/RahulPatnaik/Langchain-Builder.git
cd Langchain-Builder
```
1.  Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
1.  Install dependencies:
```bash
pip install -r requirements.txt
```

1.  Configure environment:
```
cp .env.example .env
```
# Edit .env with your API keys and settings

🔧 Configuration
----------------

Configure your service by editing the `.env` file:


# Core Settings
```
API_KEY="your-secure-key"
LOG_LEVEL="INFO"
```

# LLM Providers
```
OPENAI_API_KEY="sk-your-key"
GEMINI_API_KEY="your-gemini-key"
GROQ_API_KEY="gsk-your-key"
```

# Vector Store
```
VECTORSTORE_PATH="./chroma_db"
```

# Redis
```
REDIS_URL="redis://localhost:6379/0"
```

🏃 Running the Service
----------------------

### Development Mode
```
uvicorn app.main:app --reload --port 8000
```

### Production Mode

```
uvicorn app.main:app --workers 4 --port 8000
```


📚 API Guide
------------

### 🔐 Authentication

Include API key in headers:

```
curl -H "X-API-KEY: your-secure-key" ...
```
### 💬 Chat Endpoints

**Get Completion**

```
curl -X POST http://localhost:8000/api/v1/chat/completion\
  -H "X-API-KEY: your-key"\
  -d '{
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ]
  }'
```

**Stream Responses**

```
curl -N -X POST http://localhost:8000/api/v1/chat/stream\
  -H "X-API-KEY: your-key"\
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me a joke"}
    ]
  }'
```

### 📄 Document Processing

**Upload Document**
```
curl -X POST http://localhost:8000/api/v1/documents/index\
  -H "X-API-KEY: your-key"\
  -F "file=@document.pdf"\
  -F "collection_name=research"
```

**Query Documents**

```
curl -X POST http://localhost:8000/api/v1/documents/query\
  -H "X-API-KEY: your-key"\
  -d '{
    "query": "What are the main findings?",
    "collection_name": "research",
    "top_k": 3
  }'
```

🛠️ Advanced Features
---------------------

### Monitoring

```
curl http://localhost:8000/metrics
```

### Tracing

Configure OpenTelemetry in `.env`:

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
OTEL_SERVICE_NAME=langchain-service
```

### S3 Storage

Enable in `.env`:

```
S3_ENABLED=True
S3_BUCKET_NAME=my-bucket
S3_ACCESS_KEY=your-key
S3_SECRET_KEY=your-secret
```