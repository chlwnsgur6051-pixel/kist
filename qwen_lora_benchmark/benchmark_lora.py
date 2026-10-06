import argparse
import json
import os
import platform
import random
import socket
import time

import torch
import transformers
import peft

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def gb(x):
    return x / (1024 ** 3)


def load_dolly_records(dataset_path, max_records, seed):
    records = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            instruction = obj.get("instruction", "").strip()
            context = obj.get("context", "").strip()
            response = obj.get("response", "").strip()

            if instruction and response:
                records.append({
                    "instruction": instruction,
                    "context": context,
                    "response": response,
                })

    rng = random.Random(seed)
    rng.shuffle(records)

    return records[:max_records]


def build_training_sequences(
    records,
    tokenizer,
    seq_len,
):
    """
    Dolly examples -> packed fixed-length sequences.

    Input:
        ### Instruction:
        ...

        ### Context:
        ...

        ### Response:
        ...

    Loss is calculated only on response tokens.
    Instruction/context tokens receive label -100.
    """

    all_input_ids = []
    all_labels = []

    eos_id = tokenizer.eos_token_id

    if eos_id is None:
        raise RuntimeError("Tokenizer has no EOS token.")

    for record in records:

        if record["context"]:
            prompt = (
                "### Instruction:\n"
                f"{record['instruction']}\n\n"
                "### Context:\n"
                f"{record['context']}\n\n"
                "### Response:\n"
            )
        else:
            prompt = (
                "### Instruction:\n"
                f"{record['instruction']}\n\n"
                "### Response:\n"
            )

        response = record["response"]

        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        response_ids = tokenizer.encode(
            response,
            add_special_tokens=False,
        )

        input_ids = (
            prompt_ids
            + response_ids
            + [eos_id]
        )

        labels = (
            [-100] * len(prompt_ids)
            + response_ids
            + [eos_id]
        )

        all_input_ids.extend(input_ids)
        all_labels.extend(labels)

    sequences = []

    total_length = len(all_input_ids)

    for start in range(0, total_length - seq_len + 1, seq_len):
        end = start + seq_len

        ids = all_input_ids[start:end]
        labels = all_labels[start:end]

        # response token이 하나라도 있는 sequence만 사용
        if any(label != -100 for label in labels):
            sequences.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
                )
            )

    return sequences


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=20)

    parser.add_argument("--lora_rank", type=int, default=16)

    parser.add_argument("--dataset_records", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError(
            "This benchmark currently fixes batch_size=1 "
            "for fair GPU comparison."
        )

    result = {
        "hostname": socket.gethostname(),

        "model_name": args.model_name,
        "model_path": args.model_path,

        "dataset": "Databricks Dolly 15k",
        "dataset_path": args.dataset_path,
        "dataset_records": args.dataset_records,
        "dataset_seed": args.seed,

        "seq_len": args.seq_len,
        "batch_size": args.batch_size,

        "lora_rank": args.lora_rank,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],

        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,

        "quantization": False,
        "precision": "bf16",
        "gradient_checkpointing": False,

        "optimizer": "AdamW",
        "learning_rate": 2e-4,

        "status": "unknown",
    }

    try:

        # ------------------------------------------------------------
        # CUDA check
        # ------------------------------------------------------------

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available.")

        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 is not supported on this GPU."
            )

        device = torch.device("cuda:0")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)

        result.update({
            "gpu_name": gpu_name,
            "gpu_total_memory_gb": gb(
                gpu_props.total_memory
            ),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
            "python_version": platform.python_version(),
        })

        print("=" * 70)

        print("Host:", result["hostname"])
        print("GPU:", gpu_name)
        print(
            "GPU VRAM:",
            f"{result['gpu_total_memory_gb']:.2f} GB"
        )

        print("Model:", args.model_name)
        print("Dataset: Databricks Dolly 15k")
        print("Sequence length:", args.seq_len)
        print("Batch size:", args.batch_size)

        print("=" * 70)

        # ------------------------------------------------------------
        # Tokenizer
        # ------------------------------------------------------------

        print("\n[1] Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            local_files_only=True,
        )

        # ------------------------------------------------------------
        # Dataset
        # ------------------------------------------------------------

        print("[2] Loading Dolly dataset...")

        records = load_dolly_records(
            args.dataset_path,
            args.dataset_records,
            args.seed,
        )

        print("Loaded records:", len(records))

        sequences = build_training_sequences(
            records,
            tokenizer,
            args.seq_len,
        )

        print(
            "Packed 2048-token sequences:",
            len(sequences)
        )

        required_sequences = (
            args.warmup_steps
            + args.measure_steps
        )

        if len(sequences) < required_sequences:
            raise RuntimeError(
                f"Not enough packed sequences. "
                f"Need {required_sequences}, "
                f"but only have {len(sequences)}."
            )

        result["packed_sequences"] = len(sequences)

        # ------------------------------------------------------------
        # Model
        # ------------------------------------------------------------

        print("\n[3] Loading base model...")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

        model.to(device)

        model.config.use_cache = False

        # ------------------------------------------------------------
        # LoRA
        # ------------------------------------------------------------

        print("[4] Applying LoRA...")

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(
            model,
            lora_config,
        )

        model.train()

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        result["trainable_params"] = trainable_params
        result["total_params"] = total_params
        result["trainable_percent"] = (
            100.0
            * trainable_params
            / total_params
        )

        model.print_trainable_parameters()

        torch.cuda.synchronize()

        result["model_loaded_allocated_gb"] = gb(
            torch.cuda.memory_allocated()
        )

        result["model_loaded_reserved_gb"] = gb(
            torch.cuda.memory_reserved()
        )

        # ------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------

        optimizer = torch.optim.AdamW(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            lr=2e-4,
        )

        # ------------------------------------------------------------
        # One actual training step
        # ------------------------------------------------------------

        def train_step(sequence):

            cpu_input_ids, cpu_labels = sequence

            input_ids = (
                cpu_input_ids
                .unsqueeze(0)
                .to(device)
            )

            labels = (
                cpu_labels
                .unsqueeze(0)
                .to(device)
            )

            attention_mask = torch.ones_like(
                input_ids
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            return loss.detach()

        # ------------------------------------------------------------
        # Warm-up
        # ------------------------------------------------------------

        print(
            f"\n[5] Warm-up: "
            f"{args.warmup_steps} steps"
        )

        for i in range(args.warmup_steps):

            loss = train_step(
                sequences[i]
            )

            torch.cuda.synchronize()

            print(
                f"warmup "
                f"{i + 1}/"
                f"{args.warmup_steps} "
                f"loss={loss.item():.4f}"
            )

        torch.cuda.synchronize()

        # optimizer state까지 생성된 이후
        # peak statistics 초기화
        torch.cuda.reset_peak_memory_stats()

        # ------------------------------------------------------------
        # Measurement
        # ------------------------------------------------------------

        print(
            f"\n[6] Measurement: "
            f"{args.measure_steps} steps"
        )

        start = time.perf_counter()

        losses = []

        start_index = args.warmup_steps

        for i in range(args.measure_steps):

            sequence = sequences[
                start_index + i
            ]

            loss = train_step(sequence)

            losses.append(loss.item())

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        peak_allocated = (
            torch.cuda.max_memory_allocated()
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
        )

        total_tokens = (
            args.measure_steps
            * args.batch_size
            * args.seq_len
        )

        result.update({
            "status": "success",

            "elapsed_sec": elapsed,

            "sec_per_step":
                elapsed / args.measure_steps,

            "tokens_per_sec":
                total_tokens / elapsed,

            "peak_allocated_gb":
                gb(peak_allocated),

            "peak_reserved_gb":
                gb(peak_reserved),

            "first_measure_loss":
                losses[0],

            "last_measure_loss":
                losses[-1],

            "mean_measure_loss":
                sum(losses) / len(losses),
        })

        print("\n" + "=" * 70)

        print("RESULT")

        print("=" * 70)

        print(
            f"Elapsed        : "
            f"{elapsed:.3f} sec"
        )

        print(
            f"Sec / step     : "
            f"{result['sec_per_step']:.3f}"
        )

        print(
            f"Tokens / sec   : "
            f"{result['tokens_per_sec']:.2f}"
        )

        print(
            f"Peak allocated : "
            f"{result['peak_allocated_gb']:.2f} GB"
        )

        print(
            f"Peak reserved  : "
            f"{result['peak_reserved_gb']:.2f} GB"
        )

        print(
            f"Mean loss      : "
            f"{result['mean_measure_loss']:.4f}"
        )

    except torch.cuda.OutOfMemoryError as e:

        result["status"] = "OOM"
        result["error"] = str(e)

        print("\nCUDA OUT OF MEMORY")

    except Exception as e:

        result["status"] = "error"
        result["error"] = repr(e)

        print("\nERROR:", repr(e))

    finally:

        os.makedirs(
            os.path.dirname(args.output) or ".",
            exist_ok=True,
        )

        with open(
            args.output,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\nResult saved to:",
            args.output,
        )


if __name__ == "__main__":
    main()
