# LLM Edge Finetuning

Using serverless to fine-tune an LLM for the edge

## Setup

```
python -m venv ~/venvs/llm-edge-finetuning
source ~/venvs/llm-edge-finetuning/bin/activate
pip install -e .
```

## Usage

Run the fine-tuning job locally using a small model for testing:

```
unionai run llm_edge_finetuning/workflows.py train_workflow --config config/pythia_70b_deduped.json
```

Run the fine-tuning job on Union serverless:

```
unionai run --copy-all --remote workflows.py train_workflow --config config/phi_3_mini_128k_instruct.json
```

Change the `--config` input to one of the following files in the `config`
directory to fine-tune a larger model:

- `config/phi_3_mini_128k_instruct.json` 
- `config/codellama_7b_hf.json`
- `config/llama_3_8b_instruct.json`

## Todo:

- fine-tune with phi3 on serverless
- convert model to gguf (see https://www.substratus.ai/blog/converting-hf-model-gguf-model/)
- publish model to huggingface
