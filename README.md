# Image Analysis API

A secure image analysis API built with FastAPI and Ollama's LLaVA model that performs detailed safety assessments of uploaded images.

## Overview

This project implements an API service that:
- Analyzes uploaded images for security concerns
- Detects presence of people and weapons
- Provides detailed scene descriptions in Brazilian Portuguese
- Runs in a containerized environment with Docker

## Tech Stack

- **FastAPI** - Web framework for building the API
- **Ollama** - Local LLM runtime
- **LLaVA** - Multimodal LLM for image analysis
- **Docker** - Containerization platform
- **Python Pillow** - Image processing

## Getting Started

### Prerequisites

- Docker and Docker Compose installed on your system
- At least 8GB of RAM recommended
- GPU support (optional but recommended)

### Installation & Running

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Start the services using Docker Compose:
```bash
docker compose up --build
```

This will:
- Build and start the FastAPI application
- Pull and run the Ollama service with LLaVA model
- Set up the required networking

### Verifying Installation

1. Check if containers are running:
```bash
docker ps -a
```

2. Access the API documentation:
- OpenAPI docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Usage

### Analyze Image Endpoint

`POST /analyze-image/`

Upload an image for analysis. The API will return:
- Detailed scene description (in Portuguese)
- Presence of people (true/false)
- Presence of weapons (true/false)

Example response:
```json
{
  "image_context": "Descrição detalhada da cena em português",
  "has_weapon": false,
  "has_people": true
}
```

## Project Structure

```
.
├── fastapi/
│   ├── app.py              # Main FastAPI application
│   ├── py_utils.py         # Utility functions
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         
├── ollama/
│   ├── Dockerfile
│   └── pull-llava7b.sh    # LLaVA model setup script
└── compose.yml            # Docker Compose configuration
```

## Development

To enter the FastAPI container for debugging:
```bash
docker exec -it <container_name> bash
```

To check available Ollama models:
```bash
docker exec -it <ollama_container_name> ollama list
```

## License

[MIT License](LICENSE)
