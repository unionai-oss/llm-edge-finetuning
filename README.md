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
unionai run --copy-all --remote llm_edge_finetuning/workflows.py train_workflow --config config/phi_3_mini_128k_instruct.json
```

Change the `--config` input to one of the following files in the `config`
directory to fine-tune a larger model:

- `config/phi_3_mini_128k_instruct.json` 
- `config/codellama_7b_hf.json`
- `config/llama_3_8b_instruct.json`

## Local Inference

Clone the llama.cpp repo:

```bash
git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
```

Create an inference virtual environment:

```bash
python -m venv ~/venvs/llama-cpp
source ~/venvs/llama-cpp/bin/activate
pip install -r ~/llama.cpp/requirements.txt
```

Download the fine-tuned model:

```bash
huggingface-cli download \
   unionai/Phi-3-mini-128k-instruct-news-headlines \
    --local-dir ~/models/phi-3-mini-128k-instruct-news-headlines
```

Convert to GGUF format:

```bash
python ~/llama.cpp/convert-hf-to-gguf.py \
    ~/models/phi-3-mini-128k-instruct-news-headlines \
    --outfile ~/models/phi3_mini_128k_instruct_news.gguf \
    --outtype q8_0
```

Create the model in Ollama using the provided `Modelfile`:

```bash
ollama create phi3_mini_128k_instruct_news -f Modelfile
```

Interact with the model locally

```bash
ollama run phi3_mini_128k_instruct_news
```
