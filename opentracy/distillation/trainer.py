"""
Training Service — local SFT / Online J-BOND fine-tuning.

This module is designed to be run as a subprocess from the pipeline orchestrator,
because importing torch/unsloth is heavy and requires GPU.

Usage (subprocess):
    python -m opentracy.distillation.trainer /path/to/config.json

The config JSON must contain: job_id, tenant_id, base_model, dataset_path,
output_dir, plus optional hyperparameters.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from opentracy._env import env

logger = logging.getLogger(__name__)

def _data_dir() -> Path:
    """Resolve the data dir at call time — ``ot.distill()`` sets
    OPENTRACY_DATA_DIR via a context manager AFTER this module has already
    been imported, so capturing at import-time freezes the stale default.
    """
    return Path(env("DATA_DIR", "data"))


async def start_training(
    job_id: str,
    tenant_id: str,
    config: dict[str, Any],
    curated_path: str,
) -> dict[str, Any]:
    """
    Launch the training subprocess and wait for it to complete.

    Args:
        job_id: Distillation job ID
        tenant_id: Tenant ID
        config: Distillation config dict
        curated_path: Path to curated.jsonl

    Returns:
        Dict with output_dir and status
    """
    import asyncio

    job_dir = _data_dir() / "distillation" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    output_dir = job_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build subprocess config
    # Detect GPU VRAM and adjust defaults for low-memory GPUs
    _low_vram = False
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            vram_gb = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
            _low_vram = vram_gb < 4.0
            logger.info("GPU VRAM: %.2f GiB (low_vram=%s)", vram_gb, _low_vram)
    except Exception:
        pass

    _default_batch = 1 if _low_vram else 2
    _default_seq_len = 512 if _low_vram else 2048
    _default_grad_accum = 8 if _low_vram else 4

    train_config = {
        "job_id": job_id,
        "tenant_id": tenant_id,
        "base_model": config.get("base_model", "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"),
        "dataset_path": curated_path,
        "output_dir": str(output_dir),
        "training_mode": config.get("training_mode", "sft"),
        "max_steps": config.get("training_steps", 500),
        "learning_rate": config.get("learning_rate", 2e-4),
        "per_device_batch_size": config.get("batch_size", _default_batch),
        "grad_accum_steps": config.get("grad_accum_steps", _default_grad_accum),
        "max_seq_length": config.get("max_seq_length", _default_seq_len),
        "warmup_steps": config.get("warmup_steps", 5),
        # BOND params
        "bond_beta": config.get("bond_beta", 0.5),
        "bond_gamma": config.get("bond_gamma", 0.1),
        "bond_ema_decay": config.get("bond_ema_decay", 0.99),
        "temperature": config.get("temperature", 0.8),
        # ClickHouse connection (inherited from env)
        "ch_host": env("CH_HOST", "localhost"),
        "ch_port": env("CH_PORT", "8123"),
    }

    config_path = job_dir / "train_config.json"
    config_path.write_text(json.dumps(train_config, indent=2))

    logger.info("Launching training subprocess for job %s", job_id)

    # Inherit env + skip TensorFlow imports (we only use PyTorch).
    # NOTE: deliberately NOT named ``env`` — that shadows the module-level
    # ``env()`` import above and makes any earlier lookup (``env("CH_HOST")``)
    # raise ``UnboundLocalError`` because Python scopes ``env`` as local for
    # the whole function as soon as it sees an assignment anywhere in it.
    proc_env = {
        **os.environ,
        "TRANSFORMERS_NO_TF": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    # Pull HuggingFace token from secrets (for gated repos)
    try:
        from ..storage.secrets import get_secret
        hf_token = get_secret("huggingface")
        if hf_token:
            proc_env["HF_TOKEN"] = hf_token
            proc_env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    except Exception:
        pass

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "opentracy.distillation.trainer",
        str(config_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=proc_env,
    )

    # Stream output
    output_lines: list[str] = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="replace").rstrip()
        output_lines.append(decoded)
        logger.info("[TRAIN] %s", decoded)

    await proc.wait()

    if proc.returncode != 0:
        error_msg = "\n".join(output_lines[-20:])
        raise RuntimeError(f"Training failed (exit {proc.returncode}): {error_msg}")

    return {
        "output_dir": str(output_dir),
        "status": "completed",
    }



def _run_training(config_path: str) -> None:
    """Execute training. Called as __main__ in a subprocess."""
    # Disable torch dynamo (incompatible with unsloth patches)
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    config = json.loads(Path(config_path).read_text())

    job_id = config["job_id"]
    tenant_id = config["tenant_id"]
    base_model = config["base_model"]
    dataset_path = config["dataset_path"]
    output_dir = config["output_dir"]
    training_mode = config.get("training_mode", "sft")
    max_steps = config.get("max_steps", 500)
    learning_rate = config.get("learning_rate", 2e-4)
    batch_size = config.get("per_device_batch_size", 2)
    grad_accum = config.get("grad_accum_steps", 4)
    max_seq_length = config.get("max_seq_length", 2048)
    warmup_steps = config.get("warmup_steps", 5)

    print(f"[TRAIN] job={job_id} model={base_model} mode={training_mode}")
    print(f"[TRAIN] steps={max_steps} lr={learning_rate} batch={batch_size}")

    # Import heavy deps only in subprocess
    import torch
    from datasets import load_dataset

    print(f"[TRAIN] PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.2f} GiB")
        if vram_gb < 4.0:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            torch.cuda.empty_cache()
            print("[TRAIN] Low VRAM detected — enabled expandable_segments & cleared cache")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[TRAIN] ERROR: unsloth not installed. Install with: pip install unsloth")
        sys.exit(1)

    # Load dataset
    print(f"[TRAIN] Loading dataset from {dataset_path}")
    raw_ds = load_dataset("json", data_files=dataset_path, split="train")
    print(f"[TRAIN] Dataset: {len(raw_ds)} examples")

    # Load model
    print(f"[TRAIN] Loading model: {base_model}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            device_map="sequential",
        )
    except Exception as e:
        err_str = str(e)
        is_gated = (
            "401" in err_str or "403" in err_str
            or "gated" in err_str.lower()
            or "authorization" in err_str.lower()
            or "authenticated" in err_str.lower()
            or "access to model" in err_str.lower()
            or "is not a valid model identifier" in err_str.lower()
            or "is not a local folder" in err_str.lower()
            or "must be authenticated" in err_str.lower()
        )
        if is_gated:
            raise RuntimeError(
                f"Could not load base model '{base_model}' — HuggingFace access required.\n\n"
                f"This model is gated, private, or requires authentication. To unlock it:\n"
                f"  1. Visit https://huggingface.co/{base_model} and accept the model's license\n"
                f"  2. Create a 'read' token at https://huggingface.co/settings/tokens\n"
                f"  3. In the OpenTracy UI, go to Settings → Integrations and add your "
                f"HuggingFace API key (provider: 'HuggingFace')\n"
                f"  4. Re-run the training job\n\n"
                f"Original error: {err_str[:200]}"
            ) from e
        raise
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    print("[TRAIN] Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # ClickHouse metrics callback
    ch_callback = _ClickHouseMetricsCallback(job_id, tenant_id, config)

    if training_mode == "bond":
        _run_bond_training(
            model, tokenizer, raw_ds, config, ch_callback,
            max_steps=max_steps, learning_rate=learning_rate,
            batch_size=batch_size, grad_accum=grad_accum,
            warmup_steps=warmup_steps, supports_bf16=supports_bf16,
        )
    else:
        _run_sft_training(
            model, tokenizer, raw_ds, config, ch_callback,
            max_steps=max_steps, learning_rate=learning_rate,
            batch_size=batch_size, grad_accum=grad_accum,
            warmup_steps=warmup_steps, max_seq_length=max_seq_length,
            supports_bf16=supports_bf16,
        )

    # Save adapter
    print(f"[TRAIN] Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("[TRAIN] Training completed successfully!")


def _run_sft_training(
    model, tokenizer, dataset, config, ch_callback,
    *, max_steps, learning_rate, batch_size, grad_accum,
    warmup_steps, max_seq_length, supports_bf16,
):
    """Standard SFT training on curated data."""
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Convert dataset to chat-template format if it has prompt/response columns,
    # otherwise fall back to raw "text" field.
    has_messages = "messages" in dataset.column_names
    has_prompt_response = "prompt" in dataset.column_names and "response" in dataset.column_names

    if has_messages:
        # Already in messages format (tool_call traces) — apply chat template
        def apply_chat_template(example):
            messages = example["messages"]
            if isinstance(messages, str):
                import json as _json
                messages = _json.loads(messages)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            example["text"] = text
            return example
        dataset = dataset.map(apply_chat_template)
        print(f"[TRAIN] Applied chat template to {len(dataset)} examples (messages format)")

    elif has_prompt_response:
        # Convert prompt/response to chat template
        def format_as_chat(example):
            messages = []
            system = example.get("system_prompt") or example.get("system", "")
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": example["prompt"]})

            # Check for tool_calls in response
            tool_calls_str = example.get("tool_calls", "")
            if tool_calls_str:
                import json as _json
                try:
                    tool_calls = _json.loads(tool_calls_str) if isinstance(tool_calls_str, str) else tool_calls_str
                    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                except (ValueError, TypeError):
                    messages.append({"role": "assistant", "content": example["response"]})
            else:
                messages.append({"role": "assistant", "content": example["response"]})

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            example["text"] = text
            return example
        dataset = dataset.map(format_as_chat)
        print(f"[TRAIN] Converted {len(dataset)} prompt/response pairs to chat template")

    else:
        # Raw text field — just ensure EOS
        def add_eos(example):
            text = example.get("text", "")
            if not text.endswith(tokenizer.eos_token):
                example["text"] = f"{text}{tokenizer.eos_token}"
            return example
        dataset = dataset.map(add_eos)
        print(f"[TRAIN] Using raw text field for {len(dataset)} examples")

    # Detect low VRAM for extra memory optimizations
    _low_vram_train = False
    if torch.cuda.is_available():
        _vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        _low_vram_train = _vram < 4.0

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=(not supports_bf16) and torch.cuda.is_available(),
        bf16=supports_bf16,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=not _low_vram_train,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    trainer.add_callback(ch_callback)

    print("[TRAIN] Starting SFT training...")
    trainer.train()
    print("[TRAIN] SFT training complete!")


def _run_bond_training(
    model, tokenizer, dataset, config, ch_callback,
    *, max_steps, learning_rate, batch_size, grad_accum,
    warmup_steps, supports_bf16,
):
    """Online J-BOND training with reward model."""
    import copy
    import torch
    import torch.nn.functional as F
    from transformers import Trainer, TrainingArguments, TrainerCallback

    bond_beta = config.get("bond_beta", 0.5)
    bond_gamma = config.get("bond_gamma", 0.1)
    bond_ema_decay = config.get("bond_ema_decay", 0.99)
    bond_lr = learning_rate if learning_rate < 1e-4 else 3e-6

    # Create frozen anchor model
    print("[BOND] Creating anchor model (frozen EMA copy)...")
    anchor_model = copy.deepcopy(model)
    for p in anchor_model.parameters():
        p.requires_grad = False
    anchor_model.eval()

    # Reward scorer (LLM-as-Judge via Go engine API)
    from ..evals_common.model_invoker import ModelInvoker
    invoker = ModelInvoker()
    judge_model = config.get("judge_model", "openai/gpt-4o-mini")

    def score_batch(prompts, responses):
        from .curation import _score_response
        scores = []
        for p, r in zip(prompts, responses):
            try:
                s = _score_response(invoker, judge_model, p, r)
            except Exception:
                s = 0.5
            scores.append(s)
        return scores

    # Prompt collator
    class _PromptCollator:
        def __call__(self, features):
            prompts = [f.get("prompt", "") for f in features if f.get("prompt")]
            return {"prompts": prompts}

    gen_config = {
        "max_new_tokens": config.get("bond_max_new_tokens", 512),
        "temperature": config.get("temperature", 0.8),
        "do_sample": True,
        "top_p": 0.9,
    }

    # Custom BOND trainer using Trainer base
    class _OnlineBONDTrainer(Trainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._bond_metrics = {}

        def _generate(self, prompts, mdl, n=1):
            mdl.eval()
            responses = []
            with torch.no_grad():
                for prompt in prompts:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True)
                    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
                    for _ in range(n):
                        out = mdl.generate(**inputs, **gen_config, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                        prompt_len = inputs["input_ids"].shape[1]
                        resp = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
                        responses.append(resp)
            mdl.train()
            return responses

        def _log_probs(self, mdl, prompts, responses):
            lps = []
            for prompt, response in zip(prompts, responses):
                full = f"{prompt}{response}"
                full_enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
                prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                ids = full_enc["input_ids"].to(mdl.device)
                mask = full_enc["attention_mask"].to(mdl.device)
                plen = prompt_enc["input_ids"].shape[1]
                out = mdl(input_ids=ids, attention_mask=mask)
                logits = out.logits[:, plen - 1:-1, :]
                labels = ids[:, plen:]
                lp = F.log_softmax(logits, dim=-1)
                token_lps = lp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                lps.append(token_lps.sum())
            return torch.stack(lps)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            prompts = inputs.get("prompts", [])
            bs = len(prompts)
            if bs == 0:
                zero = torch.tensor(0.0, device=model.device, requires_grad=True)
                return (zero, None) if return_outputs else zero

            # Generate from policy and anchor
            policy_resp = self._generate(prompts, model, n=1)
            anchor_resp_1 = self._generate(prompts, anchor_model, n=1)
            anchor_resp_2 = self._generate(prompts, anchor_model, n=1)

            # Score
            r_pol = score_batch(prompts, policy_resp)
            r_anc1 = score_batch(prompts, anchor_resp_1)
            r_anc2 = score_batch(prompts, anchor_resp_2)

            # Best-of-2 and J-BOND penalty
            penalty = -torch.log(torch.tensor(16.0))
            best_anchor = []
            r_jbond = []
            for i in range(bs):
                ba = anchor_resp_1[i] if r_anc1[i] >= r_anc2[i] else anchor_resp_2[i]
                best_anchor.append(ba)
                if r_pol[i] < min(r_anc1[i], r_anc2[i]):
                    r_jbond.append(penalty)
                else:
                    r_jbond.append(torch.tensor(0.0))
            r_jbond_t = torch.stack(r_jbond).to(model.device)

            # Forward KL (SFT on best anchor)
            loss_fw = torch.tensor(0.0, device=model.device)
            cnt = 0
            for p, r in zip(prompts, best_anchor):
                if not r:
                    continue
                full = f"{p}{r}{tokenizer.eos_token}"
                full_enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
                prompt_enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=2048)
                ids = full_enc["input_ids"].to(model.device)
                mask = full_enc["attention_mask"].to(model.device)
                labels = ids.clone()
                labels[:, :prompt_enc["input_ids"].shape[1]] = -100
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss_fw = loss_fw + out.loss
                cnt += 1
            loss_fw = loss_fw / max(cnt, 1)

            # Backward KL (policy gradient)
            log_probs = self._log_probs(model, prompts, policy_resp)
            loss_bw = -(log_probs * r_jbond_t).mean()

            # KL regularization
            policy_lp = self._log_probs(model, prompts, policy_resp)
            with torch.no_grad():
                anchor_lp = self._log_probs(anchor_model, prompts, policy_resp)
            loss_kl = (policy_lp - anchor_lp).mean()

            loss = (1 - bond_beta) * loss_fw + bond_beta * loss_bw + bond_gamma * loss_kl

            self._bond_metrics = {
                "loss_forward_kl": loss_fw.item(),
                "loss_backward_kl": loss_bw.item(),
                "loss_kl_reg": loss_kl.item(),
                "reward_policy_mean": sum(r_pol) / max(len(r_pol), 1),
                "reward_anchor_mean": sum(max(a1, a2) for a1, a2 in zip(r_anc1, r_anc2)) / max(bs, 1),
                "jbond_penalties": sum(1 for r in r_jbond if r < 0) / max(bs, 1),
            }
            return (loss, None) if return_outputs else loss

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            eta = 1 - bond_ema_decay
            with torch.no_grad():
                for (n, ap), (_, pp) in zip(anchor_model.named_parameters(), model.named_parameters()):
                    if ap.dtype in (torch.float32, torch.float16, torch.bfloat16):
                        ap.data.mul_(bond_ema_decay).add_(pp.data, alpha=eta)
            return loss

        def log(self, logs, *args, **kwargs):
            if self._bond_metrics:
                logs.update(self._bond_metrics)
                self._bond_metrics = {}
            super().log(logs, *args, **kwargs)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=bond_lr,
        fp16=(not supports_bf16) and torch.cuda.is_available(),
        bf16=supports_bf16,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = _OnlineBONDTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=_PromptCollator(),
    )
    trainer.add_callback(ch_callback)

    print("[BOND] Starting Online J-BOND training...")
    trainer.train()
    print("[BOND] J-BOND training complete!")



try:
    from transformers import TrainerCallback as _CallbackBase
except ImportError:
    _CallbackBase = object  # type: ignore[misc,assignment]


class _ClickHouseMetricsCallback(_CallbackBase):  # type: ignore[misc]
    """TrainerCallback that writes per-step metrics to ClickHouse."""

    def __init__(self, job_id: str, tenant_id: str, config: dict):
        self.job_id = job_id
        self.tenant_id = tenant_id
        self._last_progress_update = 0

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"[METRICS] Training started: {args.max_steps} steps")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        if loss is None:
            return

        import time

        # Write metric to ClickHouse
        try:
            from . import repository as _repo
            metric = {
                "job_id": self.job_id,
                "tenant_id": self.tenant_id,
                "step": state.global_step,
                "epoch": float(logs.get("epoch", 0)),
                "loss": float(loss),
                "learning_rate": float(logs.get("learning_rate", 0)),
                "reward_policy_mean": float(logs.get("reward_policy_mean", 0)),
                "reward_anchor_mean": float(logs.get("reward_anchor_mean", 0)),
                "kl_penalty": float(logs.get("kl_penalty", 0)),
                "bond_loss": float(logs.get("bond_loss", 0)),
                "anchor_loss": float(logs.get("anchor_loss", 0)),
                "jeffreys_kl": float(logs.get("jeffreys_kl", 0)),
                "loss_forward_kl": float(logs.get("loss_forward_kl", 0)),
                "loss_backward_kl": float(logs.get("loss_backward_kl", 0)),
                "loss_kl_reg": float(logs.get("loss_kl_reg", 0)),
                "jbond_penalties": float(logs.get("jbond_penalties", 0)),
                "reward_improvement": float(logs.get("reward_improvement", 0)),
            }
            _repo.insert_metric(metric)
        except Exception as e:
            print(f"[METRICS] Failed to write metric: {e}")

        # Update job progress (throttled)
        now = time.time()
        if now - self._last_progress_update >= 10:
            self._last_progress_update = now
            try:
                from . import repository as _repo
                pct = min(99, int(state.global_step / max(state.max_steps, 1) * 100))
                job = _repo.get_job(self.tenant_id, self.job_id)
                if job:
                    progress = job.get("progress", {})
                    progress["training"] = {
                        "status": "running",
                        "progress": pct,
                        "training_loss": round(float(loss), 6),
                        "current_step": state.global_step,
                        "total_steps": state.max_steps,
                        "training_epoch": round(float(logs.get("epoch", 0)), 4),
                    }
                    _repo.update_job(self.tenant_id, self.job_id, {"progress": progress})
            except Exception as e:
                print(f"[METRICS] Failed to update progress: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        try:
            from . import repository as _repo
            job = _repo.get_job(self.tenant_id, self.job_id)
            if job:
                progress = job.get("progress", {})
                progress["training"] = {
                    "status": "completed",
                    "progress": 100,
                    "current_step": state.max_steps,
                    "total_steps": state.max_steps,
                }
                _repo.update_job(self.tenant_id, self.job_id, {"progress": progress})
        except Exception as e:
            print(f"[METRICS] Failed to update on_train_end: {e}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m opentracy.distillation.trainer <config.json>")
        sys.exit(1)
    _run_training(sys.argv[1])
