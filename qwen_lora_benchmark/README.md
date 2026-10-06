# Qwen LoRA benchmark

`benchmark_lora.py` measures short-run LoRA fine-tuning throughput and peak GPU memory for a local causal language model. It uses packed Databricks Dolly 15k instruction examples and computes loss on response tokens only.

## Run

The benchmark requires local model and Dolly JSONL files. It uses CUDA BF16, batch size 1, and defaults to 5 warm-up steps followed by 20 measured steps. Pass paths that exist on the machine running the benchmark:

```bash
python benchmark_lora.py \
  --model_path /path/to/Qwen2.5-3B \
  --model_name Qwen2.5-3B \
  --dataset_path /path/to/databricks-dolly-15k.jsonl \
  --output results/run.json
```

The script records configuration, software versions, GPU details, throughput, memory, loss, and failures in the selected JSON output. The committed `results/` files are historical records, including OOM outcomes. They preserve the paths and host metadata recorded at the time of those runs.

The recorded benchmark environment used PyTorch 2.11.0+cu128, Transformers 5.17.0, PEFT 0.21.1, and Python 3.11.16. CUDA-compatible PyTorch installation depends on the target machine; do not upgrade an existing GPU environment blindly.
