# Phase 2 Local Model Setup

Phase 2 expects an OpenAI-compatible local chat endpoint.

## Installed Paper Models

The paper runs use reproducible Ollama aliases with `num_ctx 8192`. Create them
after pulling the base models:

```bash
bash scripts/phase2_create_ollama_models.sh
```

The local model matrix is:

```text
arfa-qwen2.5-coder:7b-8k
arfa-llama3.1:8b-8k
arfa-qwen2.5-coder:14b-8k
```

The 7B coding model is the primary baseline, Llama 3.1 8B provides a cross-family comparison, and Qwen 14B tests within-family scale. All three use the same effective 8192-token context and run locally and serially because the experiment machine has 24 GB unified memory.

## Install Ollama

Install Ollama for macOS from:

```text
https://ollama.com/download/mac
```

Then pull the model:

```bash
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:14b
```

Ollama usually starts the local server automatically. If needed:

```bash
ollama serve
```

## Check The Endpoint

```bash
bash scripts/phase2_check_model.sh
```

Then run the development fixture:

```bash
bash scripts/phase2_run_baseline.sh
```

## Full Official Run

Docker must be healthy before running official InterCode tasks. Then use:

```bash
bash scripts/phase2_build_intercode_images.sh
bash scripts/phase2_validate_official_environment.sh
bash scripts/phase2_run_paper_matrix.sh
```

The matrix script is resumable and runs the models serially.
