# HR CV Analysis System

An AI-powered system for analyzing resumes (CVs) against job descriptions using advanced natural language processing and machine learning.

## Features

- **Document Processing**: Extract text from PDF and DOCX files using OCR technology
- **AI-Powered Analysis**: Leverages Ollama LLMs for intelligent CV and job description parsing
- **Experience Calculation**: Automatically calculates and enriches candidate experience data
- **Suitability Matching**: Computes compatibility scores between candidates and job requirements
- **REST API**: FastAPI-based RESTful API for easy integration
- **Containerized**: Docker support for easy deployment

## Tech Stack

- **Backend**: FastAPI (Python 3.12)
- **AI/ML**: Ollama, LangChain, PaddleOCR
- **Document Processing**: PDFPlumber, python-docx
- **Containerization**: Docker
- **Package Management**: uv
- **Logging**: Loguru

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Ollama running locally (for LLM services)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hr-cv-analysis
```

2. Build and run with Docker:
```bash
docker build -t hr-cv-analysis .
docker run -p 8000:8000 hr-cv-analysis
```

### Local Development

1. Install dependencies:
```bash
pip install uv
uv sync
```

2. Set up environment variables (create `.env` file):
```env
OLLAMA_URL=http://localhost:11434
MODEL_NAME=llama2
THINKING_MODEL=llama2:13b
```

3. Run the application:
```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

### Health Check
```bash
GET /health
```

### CV Analysis
```bash
POST /v1/analyze
Content-Type: multipart/form-data

Files:
- jd_file: Job description document (PDF/DOCX)
- cv_file: Candidate resume (PDF/DOCX)
```

**Response**: JSON with analysis results including:
- Parsed job requirements
- Candidate qualifications
- Experience calculations
- Suitability score

## Project Structure

```
├── backend/
│   ├── agents/          # AI agents for processing
│   ├── core/            # Configuration and dependencies
│   ├── middlewares/     # FastAPI middlewares
│   ├── routes/          # API endpoints
│   ├── schemas/         # Pydantic models
│   └── utils/           # Utility functions
├── Dockerfile           # Container configuration
├── pyproject.toml       # Project dependencies
└── README.md            # README file
```

## Development

### Code Quality
- **Linting**: Ruff
- **Formatting**: Black
- **Type checking**: MyPy
- **Testing**: Pytest

Run quality checks:
```bash
uv run ruff check .
uv run black .
uv run mypy .
```

### Pre-commit Hooks
```bash
pre-commit install
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

## License

[Add your license here]
