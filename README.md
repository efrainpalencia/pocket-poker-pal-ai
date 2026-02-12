# Pocket Poker Pal AI API

AI-Powered poker rules assistant equipped with RAG.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Description

Pocket Poker Pal AI API is an AI-powered assistant for poker rules and guidance. It uses Retrieval-Augmented Generation (RAG) to combine a retrieval layer (for factual source grounding) with generative models to provide accurate, context-aware answers about poker rules, hand rankings, and gameplay scenarios.

## Features

- Retrieval-Augmented Generation (RAG) for grounded answers
- Streamed chat API endpoints for interactive sessions
- Modular graph-based retrieval and chain components
- Lightweight, test-covered Python codebase

![Uses clarifies questions, retrieves datasource, and grades answers.](/graph.png "Graph Workflow Diagram")

## Project Structure

Top-level layout (important files and folders):

```
pyproject.toml
README.md
src/
	__init__.py
	main.py
	cli.py
	ingestion.py
	api/
		v1/
			routes/
				chat.py
				chat_stream.py
	graph/
		graph.py
		retrieval_debug.py
		vectorstore.py
	chains/
		classifier.py
		generation.py
		grader.py
	services/
		chat_service.py
		chat_stream_service.py
	schemas/
		chat_schema.py
	llm/
		factory.py

```

Refer to the `src` package for the main application logic and `api/v1/routes` for the HTTP endpoints.

## Quickstart

Prerequisites: Python 3.13+ and a virtual environment (see `pyproject.toml`).

1. Create and activate a virtualenv:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Unix/macOS
   .venv\Scripts\Activate.ps1 # Windows PowerShell
   ```

2. Editable install (recommended during development):
   - Upgrade packaging tools and install in editable mode (PEP 660-backed editable install):

     ```bash
     python -m pip install --upgrade pip build wheel setuptools
     python -m pip install -e .
     ```

   - If your environment doesn't support editable installs, install normally:

     ```bash
     python -m pip install .
     ```

   - Alternatively, if you prefer using a `requirements.txt`, generate it from `pyproject.toml` tools or maintain a separate `requirements.txt` and install with:

     ```bash
     python -m pip install -r requirements.txt
     ```

3. Run the application (example):

   ```bash
   python -m src.main
   ```

4. Call the API endpoints under `api/v1` (see `src/api/v1/routes`).

## API Usage

Base URL (local dev): `http://localhost:8000/api/v1`

- POST /qa — Ask a question (JSON body):

  ```bash
  curl -X POST "http://localhost:8000/api/v1/qa" \
  	-H "Content-Type: application/json" \
  	-d '{"question": "What beats a full house?", "thread_id": "optional-thread-123"}'
  ```

- POST /qa/resume — Resume a thread with a reply:

  ```bash
  curl -X POST "http://localhost:8000/api/v1/qa/resume" \
  	-H "Content-Type: application/json" \
  	-d '{"thread_id": "optional-thread-123", "reply": "Thanks — can you clarify the kicker rules?"}'
  ```

- GET /qa/stream — Streamed answer (SSE):

  ```bash
  curl -N "http://localhost:8000/api/v1/qa/stream?question=How+many+cards+in+a+hand%3F&thread_id=thread-1"
  ```

  Note: use `-N` or `--no-buffer` to stream output as Server-Sent Events.

- GET /qa/resume/stream — Stream resume replies (SSE):

  ```bash
  curl -N "http://localhost:8000/api/v1/qa/resume/stream?thread_id=thread-1&reply=I+agree"
  ```

Postman quick steps:

- POST endpoints: create a new POST request, set body type to `raw` → `JSON` and paste the JSON payload.
- GET stream endpoints: create a GET request with query parameters; enable `Follow original HTTP method` and use a console or the Postman `Send and Download` option to capture streaming SSE responses.

## Development

- Run unit tests with `pytest -q`.
- Keep styles with `black` and check typing with `mypy` (if configured).
- Add new endpoints under `src/api/v1/routes` and corresponding services in `src/services`.

## Testing

Tests live under `src` packages (e.g. `graph/tests`, `api/v1/routes/tests`). Run:

```bash
pytest -q
```

## Contributing

Contributions are welcome. Please open issues or pull requests describing changes and include tests for new behavior.

## License

None
