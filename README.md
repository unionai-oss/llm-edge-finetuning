# LLM Edge Finetuning

Using serverless to fine-tune an LLM for the edge

## Setup

```bash
python -m venv ~/venvs/llm-edge-finetuning
source ~/venvs/llm-edge-finetuning/bin/activate
pip install -e .
```

## Usage

Run the fine-tuning job locally using a small model for testing:

```bash
unionai run llm_edge_finetuning/workflows_news.py train_workflow \
    --config config/pythia_70b_deduped.json \
    --categories '["science", "technology", "business"]'
```

Run the fine-tuning job on Union serverless:

```bash
unionai run --copy-all --remote \
    llm_edge_finetuning/workflows_news.py train_workflow \
    --config config/phi_3_mini_128k_instruct.json \
    --categories '["science", "technology", "business"]'
```

Change the `--config` input to one of the following files in the `config`
directory to fine-tune a larger model:

- `config/phi_3_mini_128k_instruct.json`
- `config/codellama_7b_hf.json`
- `config/llama_3_8b_instruct.json`

## Local Inference

Download the fine-tuned model:

```bash
huggingface-cli download \
    unionai/Phi-3-mini-128k-instruct-news-headlines-gguf \
    --local-dir ~/models/phi-3-mini-128k-instruct-news-headlines-gguf
```

Create the model in Ollama using the `Modelfile` created by the workflow:

```bash
ollama create phi3-news -f Modelfile
```

Interact with the model locally

```bash
ollama run phi3-news
```

```
> What's the latest medical research relating to cripsr therapy?
```
