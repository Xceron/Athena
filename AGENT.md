# Athena Agent Guide

## Commands
- **Run app**: `uv run src/main.py` (starts FastAPI server on port 5000)
- **Build Docker**: `docker build -t athena .`
- **Run Docker**: `docker-compose up -d`
- **Install deps**: `uv sync`
- **No test commands found** - consider adding pytest

## Architecture
- **Structure**: Code organized in `/src/` folder
- **Main**: FastAPI web server with background tasks for PDF processing
- **APIs**: GET `/add_initial_tags/`, `/add_missing_tags/`, `/summarize/`
- **Integrations**: Zotero API, Claude (Anthropic), Gemini (Google)
- **Storage**: Zotero attachments mounted at `/app/zotero`
- **Config**: Environment variables (see docker-compose.yml)

## Code Structure
- **src/main.py**: FastAPI endpoints and main processing logic
- **src/llm.py**: LLMRouter class for unified AI model handling
- **src/zotero_utils.py**: All Zotero API operations
- **src/pdf_utils.py**: PDF handling utilities
- **prompts/**: System and user prompts for AI models

## Code Style
- **Python 3.12+**, uses modern type hints (`str | None`, `List[str]`)
- **Imports**: Standard library first, then third-party, then local
- **Logging**: Use `logger` instance, structured messages with item keys
- **Error handling**: Try/catch with specific error types, log before raising
- **Async**: Background tasks for long-running operations
- **Retry**: Use `@retry` decorator for API calls with exponential backoff
- **Environment**: All config via env vars, validate on startup
