# LLM Benchmark

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Run

```bash
python auto_bemchmark.py --model meta-llama/Meta-Llama-3-8B-Instruct --docker-image IMAGE_ID --input-tokens 100 --output-tokens 100 --concurrency 1
```
