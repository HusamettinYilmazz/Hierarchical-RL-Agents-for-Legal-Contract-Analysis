# Hierarchical RL Agents for Legal Contract Analysis

## Demo video

Watch a short walkthrough of the system in action:

<video controls width="100%" preload="metadata">
  <source src="assets/readme_demo/demo.mp4" type="video/x-matroska">
  Your browser does not support embedded video! You can download it here: [demo.mp4](assets/readme/demo.mp4)
</video>

This repository contains a full-stack workflow for legal contract analysis using large language models. It combines:

- a training pipeline for contract extraction and analysis using supervised fine-tuning (SFT) and GRPO,
- a Temporal-based PDF processing workflow for ingesting contract documents,
- a FastAPI client that exposes endpoints to start workflows, check status, and review contract analysis results.

The project is designed around contract datasets such as CUAD and uses an instruction-following LLM to extract structured legal information from contract text.

## Overview

The pipeline has three main parts:

1. Data preparation
   - The CUAD-style dataset is converted into prompt/answer examples for contract analysis.
   - Training examples are generated in JSONL format for SFT and GRPO.

2. Model training
   - SFT/LoRA training is used to adapt a base instruction model to contract analysis tasks.
   - GRPO is then used to further optimize the model with reward signals based on similarity and exact-match quality.

3. Inference and orchestration
   - PDFs are downloaded from cloud storage, converted to markdown, and processed through a Temporal workflow.
   - A FastAPI service exposes APIs for PDF processing and multi-document contract review.

## Repository structure

- `agent_train/`
  - `sft/` – SFT training code and dataset loading
  - `grpo/` – GRPO training code and reward computation
  - `preprocessing/` – preprocessing scripts for converting CUAD data into training examples
  - `configs/config.yml` – training and model configuration
  - `utils/` – shared config helpers

- `app/pdf-extraction/`
  - Temporal activities and workflows for:
    - downloading PDFs from S3,
    - converting PDFs to markdown,
    - calling an LLM for summarization or analysis,
    - uploading the processed output.

- `app/client_app/`
  - FastAPI server exposing endpoints for workflow execution and contract review.

- `setup/`
  - Temporal deployment helpers and Docker Compose configuration.

## Prerequisites

- Python 3.11
- Docker and Docker Compose (for running the Temporal workflow engine)
- A GPU is strongly recommended for model training

## Installation

Create and activate a Python environment:

```bash
conda create -n contract-analysis python=3.11
conda activate contract-analysis
pip install -r requirements.txt
```

## Configuration

### Training configuration

Update `agent_train/configs/config.yml` with your dataset path, model names, and output locations.

The config currently expects:

- a CUAD-style dataset at `data.dataset_path`
- a base instruction model at `model.base_model_name`
- an SFT checkpoint path at `model.sft_model_path`
- an output directory under `data.output_path`

### Environment files

Copy the example environment files before running the app:

```bash
cp app/pdf-extraction/.env.example app/pdf-extraction/.env
cp app/client_app/.env.example app/client_app/.env
cp setup/samples-server/compose/.env.example setup/samples-server/compose/.env
```

Then edit the copied `.env` files and replace placeholder values with your own credentials and paths.

The runtime configuration includes these variables:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_S3_ENDPOINT_URL`
- `S3_BUCKET`
- `TEMP_DIR`
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `OPENROUTER_BASE_URL`
- `MAX_TOKENS`
- `USE_LOCAL_MODEL`
- `LOCAL_BASE_MODEL`
- `LOCAL_ADAPTER_PATH`
- `TEMPORAL_HOST`
- `TEMPORAL_NAMESPACE`
- `TEMPORAL_PDF_PROCESS_TASK_QUEUE`

## Training the model

### 1) Supervised fine-tuning

```bash
python agent_train/sft/train_sft.py
```

This trains a LoRA-adapted model using the prompt/response dataset generated from the contract data.

### 2) GRPO training

```bash
python agent_train/grpo/train_grpo.py
```

This step uses the SFT model as a starting point and further optimizes it using GRPO with reward signals based on exact match and similarity to the expected answer.

## Running the application

### 1) Start the Temporal workflow engine

From the repository root:

```bash
docker compose -f setup/samples-server/compose/docker-compose-postgres.yml up -d
```

### 2) Start the Temporal worker

```bash
cd app/pdf-extraction
python worker.py
```

### 3) Start the FastAPI client

```bash
cd app/client_app
uvicorn main:app --reload --port 5000
```

The app will be available at `http://localhost:5000`.

## API endpoints

The FastAPI client exposes the following endpoints:

- `GET /health` – health check
- `POST /process-pdf/execute` – run PDF processing synchronously
- `POST /process-pdf/start` – start PDF processing asynchronously
- `GET /process-pdf/status/{workflow_id}` – check PDF workflow status
- `POST /contract-review/start` – start a multi-document contract review workflow
- `GET /contract-review/{workflow_id}/status` – check review workflow status
- `GET /contract-review/{workflow_id}/report` – fetch the current review report
- `POST /contract-review/{workflow_id}/assign` – assign a reviewer name
- `POST /contract-review/{workflow_id}/revise` – submit revision feedback
- `GET /contract-review/{workflow_id}/approve` – approve the current review result

Example requests are included in the FastAPI module source comments in `app/client_app/main.py`.

## Notes

- The repository includes sample contract PDFs under `app/pdf-extraction/assets/` for local testing.
- Training and workflow outputs are written to the `agent_train/outputs/` and `app/pdf-extraction/outputs/` directories.
- Because this repository uses model-based workflows and cloud storage integration, you may need to provide your own credentials and model checkpoints depending on your environment.
