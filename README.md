# 🧠 Smart Insurance Parser

A FastAPI-powered OCR & LLM pipeline that extracts structured insurance details from card images using PaddleOCR and OpenAI's GPT.

---

## 🚀 Features

- ✅ Extracts clean, structured insurance data from scanned cards or photos
- 🔐 Secured via Bearer token authentication
- 🤖 Combines PaddleOCR and GPT (OpenAI) for robust parsing
- 📦 Dockerized for easy deployment
- 📄 JSON API ready for integration

---

## 📦 Requirements

- Python 3.11+
- OpenAI API Key
- Docker (optional)
- `.env` file for secrets

---

## 🔧 Installation & Usage

### 🧪 Local (Python)

```bash
git clone https://github.com/hussnainarshad/smart-insurance-parser.git
cd smart-insurance-parser

# Install dependencies
pip install -r requirements.txt

# Run the API
```bash
python main.py

Docker
```bash
# Build the Docker image
docker build -t smart-insurance-parser .
