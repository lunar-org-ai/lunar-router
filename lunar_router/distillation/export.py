"""
Export Service — merge LoRA adapter and convert to GGUF format.

Runs as a subprocess from the pipeline orchestrator because it loads
transformers/peft for merging and calls llama.cpp for GGUF conversion.

Usage (subprocess):
    python -m lunar_router.distillation.export /path/to/config.json

Config JSON must contain: job_id, tenant_id, adapter_dir, base_model,
quantization_types, output_dir.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("LUNAR_DATA_DIR", "data"))


async def start_export(
    job_id: str,
    tenant_id: str,
    config: dict[str, Any],
    adapter_dir: str,
) -> dict[str, Any]:
    """
    Launch the export subprocess and wait for it to complete.

    Returns:
        Dict with artifacts paths and status.
    """
    import asyncio

    job_dir = DATA_DIR / "distillation" / job_id
    output_dir = job_dir / "gguf"
    output_dir.mkdir(parents=True, exist_ok=True)

    export_config = {
        "job_id": job_id,
        "tenant_id": tenant_id,
        "adapter_dir": adapter_dir,
        "base_model": config.get("base_model", ""),
        "quantization_types": config.get("quantization_types", ["q4_k_m", "q8_0"]),
        "output_dir": str(output_dir),
        "export_gguf": config.get("export_gguf", True),
        "ch_host": os.environ.get("LUNAR_CH_HOST", "localhost"),
        "ch_port": os.environ.get("LUNAR_CH_PORT", "8123"),
    }

    config_path = job_dir / "export_config.json"
    config_path.write_text(json.dumps(export_config, indent=2))

    logger.info("Launching export subprocess for job %s", job_id)

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "lunar_router.distillation.export",
        str(config_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parents[2]),
    )

    output_lines: list[str] = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="replace").rstrip()
        output_lines.append(decoded)
        logger.info("[EXPORT] %s", decoded)

    await proc.wait()

    if proc.returncode != 0:
        error_msg = "\n".join(output_lines[-20:])
        raise RuntimeError(f"Export failed (exit {proc.returncode}): {error_msg}")

    # Collect artifact paths
    artifacts: dict[str, str] = {}
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.suffix == ".gguf":
                # e.g. model-q4_k_m.gguf → key=q4_k_m
                quant = f.stem.replace("model-", "")
                artifacts[quant] = str(f)

    return {
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "status": "completed",
    }



def _resolve_base_model(model_id: str, adapter_dir: str) -> str:
    """Resolve quantized model IDs to full-precision equivalents for CPU merge."""
    original = model_id

    if "bnb-4bit" not in model_id and "bnb-8bit" not in model_id:
        return model_id

    # Try adapter_config.json first
    adapter_config_path = Path(adapter_dir) / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            cfg = json.loads(adapter_config_path.read_text())
            base = cfg.get("base_model_name_or_path", "")
            if base and "bnb" not in base.lower():
                print(f"[EXPORT] Using base_model from adapter_config: {base}")
                return base
        except Exception:
            pass

    # Strip unsloth quantization suffixes
    model_name = model_id.split("/")[-1]
    model_name = re.sub(r"-unsloth(-bnb-[48]bit)?", "", model_name)
    model_name = re.sub(r"-bnb-[48]bit", "", model_name)

    org_mappings = {
        "Qwen3": "Qwen", "Qwen2": "Qwen", "Qwen": "Qwen",
        "Llama": "meta-llama", "Mistral": "mistralai",
        "Phi": "microsoft", "Gemma": "google",
    }
    org = "unsloth"
    for prefix, hf_org in org_mappings.items():
        if model_name.startswith(prefix):
            org = hf_org
            break

    resolved = f"{org}/{model_name}"
    print(f"[EXPORT] Resolved quantized model: {original} → {resolved}")
    return resolved


def _run_export(config_path: str) -> None:
    """Execute export. Called as __main__ in a subprocess."""
    config = json.loads(Path(config_path).read_text())

    job_id = config["job_id"]
    tenant_id = config["tenant_id"]
    adapter_dir = config["adapter_dir"]
    base_model = config["base_model"]
    quant_types = config.get("quantization_types", ["q4_k_m", "q8_0"])
    output_dir = Path(config["output_dir"])
    export_gguf = config.get("export_gguf", True)

    print(f"[EXPORT] job={job_id} base={base_model}")
    print(f"[EXPORT] adapter={adapter_dir}")
    print(f"[EXPORT] quantization={quant_types}")
    print(f"[EXPORT] export_gguf={export_gguf}")

    _update_export_progress(tenant_id, job_id, 5, "merging_adapter")

    merged_dir = output_dir.parent / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    if export_gguf and _try_unsloth_gguf(adapter_dir, str(output_dir), quant_types, tenant_id, job_id):
        _update_export_progress(tenant_id, job_id, 90, "computing_results")
        artifacts = {}
        for f in output_dir.iterdir():
            if f.suffix == ".gguf":
                qt = f.stem.replace("model-", "")
                artifacts[qt] = str(f)
        _write_results(tenant_id, job_id, artifacts)
        try:
            from . import repository as _repo
            _repo.update_job(tenant_id, job_id, {"artifacts": {"gguf": artifacts}})
        except Exception as e:
            print(f"[EXPORT] WARN: Failed to update artifacts: {e}")
        _update_export_progress(tenant_id, job_id, 100, "complete")
        print(f"[EXPORT] Export complete! Artifacts: {list(artifacts.keys())}")
        return

    # Fallback: merge then convert
    resolved_model = _resolve_base_model(base_model, adapter_dir)
    _merge_adapter(resolved_model, adapter_dir, str(merged_dir))

    _update_export_progress(tenant_id, job_id, 40, "merged")

    if not export_gguf:
        print("[EXPORT] GGUF export disabled, skipping conversion")
        _update_export_progress(tenant_id, job_id, 100, "complete")
        return

    _update_export_progress(tenant_id, job_id, 50, "converting_gguf")

    f16_path = output_dir / "model-f16.gguf"

    llama_cpp_dir = _find_llama_cpp()
    if llama_cpp_dir:
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        if convert_script.exists():
            _run_cmd(["python", str(convert_script), str(merged_dir),
                       "--outfile", str(f16_path), "--outtype", "f16"])
        else:
            _convert_with_pip(str(merged_dir), str(f16_path))
    else:
        _convert_with_pip(str(merged_dir), str(f16_path))

    if not f16_path.exists():
        raise RuntimeError("Failed to create F16 GGUF model")

    print(f"[EXPORT] Created F16 model: {f16_path}")

    # 3. Quantize
    _update_export_progress(tenant_id, job_id, 70, "quantizing")

    quantize_bin = _find_quantize_binary(llama_cpp_dir)
    artifacts: dict[str, str] = {"f16": str(f16_path)}

    if quantize_bin:
        for qt in quant_types:
            qt = qt.strip().lower()
            if qt == "f16":
                continue
            out_path = output_dir / f"model-{qt}.gguf"
            print(f"[EXPORT] Quantizing to {qt}...")
            try:
                _run_cmd([str(quantize_bin), str(f16_path), str(out_path), qt.upper()])
                if out_path.exists():
                    artifacts[qt] = str(out_path)
                    print(f"[EXPORT] Created {qt}: {out_path}")
                else:
                    print(f"[EXPORT] WARN: Quantization output not found for {qt}")
            except Exception as e:
                print(f"[EXPORT] WARN: Failed to quantize {qt}: {e}")
    else:
        print("[EXPORT] WARN: llama-quantize not found, skipping quantization")

    # 4. Compute results
    _update_export_progress(tenant_id, job_id, 90, "computing_results")
    _write_results(tenant_id, job_id, artifacts)

    # 5. Update artifacts in job record
    try:
        from . import repository as _repo
        _repo.update_job(tenant_id, job_id, {"artifacts": {"gguf": artifacts}})
    except Exception as e:
        print(f"[EXPORT] WARN: Failed to update artifacts: {e}")

    _update_export_progress(tenant_id, job_id, 100, "complete")
    print(f"[EXPORT] Export complete! Artifacts: {list(artifacts.keys())}")


def _try_unsloth_gguf(
    adapter_dir: str, output_dir: str, quant_types: list[str],
    tenant_id: str, job_id: str,
) -> bool:
    """Try direct GGUF export via unsloth (no llama.cpp needed)."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[EXPORT] Unsloth not available, skipping direct GGUF export")
        return False

    try:
        from peft import PeftModel as _PeftModel

        adapter_cfg_path = Path(adapter_dir) / "adapter_config.json"
        if not adapter_cfg_path.exists():
            return False

        adapter_cfg = json.loads(adapter_cfg_path.read_text())
        base_id = adapter_cfg.get("base_model_name_or_path", "")
        if not base_id:
            return False

        load_4bit = "4bit" in base_id.lower() or "bnb" in base_id.lower()
        print(f"[EXPORT] Loading model for GGUF export: {base_id}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_id,
            load_in_4bit=load_4bit,
            dtype=None,
        )

        print(f"[EXPORT] Loading adapter from: {adapter_dir}")
        model = _PeftModel.from_pretrained(model, adapter_dir)

        out_path = Path(output_dir)
        primary_qt = None
        for qt in quant_types:
            qt = qt.strip().lower()
            if qt != "f16":
                primary_qt = qt
                break
        primary_qt = primary_qt or "q4_k_m"

        _update_export_progress(tenant_id, job_id, 30, f"exporting_gguf_{primary_qt}")
        gguf_file = out_path / f"model-{primary_qt}.gguf"
        print(f"[EXPORT] Saving GGUF ({primary_qt}) via unsloth...")
        model.save_pretrained_gguf(
            str(out_path), tokenizer,
            quantization_method=primary_qt,
        )

        # Unsloth saves as "unsloth.{Q_TYPE}.gguf" — rename to our convention
        for f in out_path.iterdir():
            if f.suffix == ".gguf" and f.name != gguf_file.name:
                f.rename(gguf_file)
                break

        if gguf_file.exists():
            size_mb = gguf_file.stat().st_size / (1024 * 1024)
            print(f"[EXPORT] Created {primary_qt}: {gguf_file} ({size_mb:.1f} MB)")
        else:
            # Maybe unsloth already named it correctly
            gguf_files = list(out_path.glob("*.gguf"))
            if gguf_files:
                print(f"[EXPORT] Created GGUF: {gguf_files[0]}")
            else:
                print("[EXPORT] WARN: No GGUF file found after unsloth export")
                return False

        # Export additional quantizations if requested
        for qt in quant_types:
            qt = qt.strip().lower()
            if qt == "f16" or qt == primary_qt:
                continue
            try:
                _update_export_progress(tenant_id, job_id, 60, f"exporting_gguf_{qt}")
                qt_file = out_path / f"model-{qt}.gguf"
                print(f"[EXPORT] Saving additional GGUF ({qt}) via unsloth...")
                model.save_pretrained_gguf(
                    str(out_path), tokenizer,
                    quantization_method=qt,
                )
                for f in out_path.iterdir():
                    if f.suffix == ".gguf" and f.name != gguf_file.name and f.name != qt_file.name:
                        f.rename(qt_file)
                        break
                if qt_file.exists():
                    size_mb = qt_file.stat().st_size / (1024 * 1024)
                    print(f"[EXPORT] Created {qt}: {qt_file} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"[EXPORT] WARN: Failed to export {qt}: {e}")

        print("[EXPORT] Unsloth GGUF export complete!")
        return True

    except Exception as e:
        print(f"[EXPORT] Unsloth GGUF export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _try_unsloth_merge(adapter_dir: str, output_dir: str) -> bool:
    """Try merging using unsloth — handles quantized models without gated repo access."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[EXPORT] Unsloth not available, skipping unsloth merge")
        return False

    try:
        from peft import PeftModel as _PeftModel

        # Read base model from adapter config
        adapter_cfg_path = Path(adapter_dir) / "adapter_config.json"
        if not adapter_cfg_path.exists():
            print("[EXPORT] No adapter_config.json found")
            return False

        adapter_cfg = json.loads(adapter_cfg_path.read_text())
        base_id = adapter_cfg.get("base_model_name_or_path", "")
        if not base_id:
            print("[EXPORT] No base_model_name_or_path in adapter config")
            return False

        load_4bit = "4bit" in base_id.lower() or "bnb" in base_id.lower()
        print(f"[EXPORT] Loading base model via unsloth: {base_id} (4bit={load_4bit})")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_id,
            load_in_4bit=load_4bit,
            dtype=None,
        )

        print(f"[EXPORT] Loading adapter from: {adapter_dir}")
        model = _PeftModel.from_pretrained(model, adapter_dir)

        print("[EXPORT] Merging (16-bit) via unsloth...")
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
        print("[EXPORT] Unsloth merge complete!")
        return True

    except Exception as e:
        print(f"[EXPORT] Unsloth merge failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _merge_adapter(base_model_id: str, adapter_dir: str, output_dir: str) -> None:
    """Merge LoRA adapter with base model."""
    print(f"[EXPORT] Merging adapter with base model: {base_model_id}")

    # Try unsloth merge first — works with quantized models and avoids gated repo issues
    if _try_unsloth_merge(adapter_dir, str(Path(output_dir))):
        return

    # Fallback: standard transformers merge (requires full-precision base model access)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[EXPORT] Loading base model (transformers fallback)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"[EXPORT] Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("[EXPORT] Merging...")
    merged = model.merge_and_unload()

    print(f"[EXPORT] Saving merged model to: {output_dir}")
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("[EXPORT] Merge complete!")


def _find_llama_cpp() -> Path | None:
    """Find llama.cpp installation."""
    candidates = [
        Path("/opt/llama.cpp"),
        Path.home() / "llama.cpp",
        Path("llama.cpp"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # Check if llama-cpp-python is installed
    try:
        import llama_cpp
        return Path(llama_cpp.__file__).parent
    except ImportError:
        pass

    return None


def _find_quantize_binary(llama_cpp_dir: Path | None) -> Path | None:
    """Find the llama-quantize binary."""
    if llama_cpp_dir:
        candidates = [
            llama_cpp_dir / "build" / "bin" / "llama-quantize",
            llama_cpp_dir / "llama-quantize",
            llama_cpp_dir / "quantize",
        ]
        for c in candidates:
            if c.exists():
                return c

    # Check PATH
    result = shutil.which("llama-quantize")
    if result:
        return Path(result)

    return None


def _convert_with_pip(model_dir: str, output_path: str) -> None:
    """Try converting using pip-installed gguf package."""
    try:
        _run_cmd(["python", "-m", "gguf.convert",
                   "--model", model_dir, "--outfile", output_path, "--outtype", "f16"])
    except Exception:
        # Fallback: try the convert script from llama-cpp-python
        try:
            import llama_cpp
            scripts_dir = Path(llama_cpp.__file__).parent.parent
            convert_script = scripts_dir / "scripts" / "convert_hf_to_gguf.py"
            if convert_script.exists():
                _run_cmd(["python", str(convert_script), model_dir,
                           "--outfile", output_path, "--outtype", "f16"])
            else:
                raise RuntimeError("No GGUF conversion tool found")
        except ImportError:
            raise RuntimeError(
                "No GGUF conversion tool found. Install llama-cpp-python or "
                "clone llama.cpp to /opt/llama.cpp"
            )


def _run_cmd(cmd: list[str], cwd: Path | str | None = None) -> None:
    """Run a shell command."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-5:]:
            print(f"  {line}")


def _update_export_progress(tenant_id: str, job_id: str, pct: int, step: str) -> None:
    try:
        from . import repository as _repo
        job = _repo.get_job(tenant_id, job_id)
        if not job:
            return
        progress = job.get("progress", {})
        progress["export"] = {
            "status": "completed" if pct >= 100 else "running",
            "progress": pct,
            "step": step,
        }
        _repo.update_job(tenant_id, job_id, {"progress": progress})
    except Exception as e:
        print(f"[EXPORT] Failed to update progress: {e}")


def _write_results(tenant_id: str, job_id: str, artifacts: dict[str, str]) -> None:
    """Write results summary to job record."""
    try:
        from . import repository as _repo

        # Get quality score from training metrics
        quality_score = 0.85
        latest_metric = _repo.get_latest_metric(job_id)
        if latest_metric:
            reward = latest_metric.get("reward_policy_mean", 0)
            if reward and float(reward) > 0:
                quality_score = max(0.0, min(1.0, float(reward)))
            else:
                loss = float(latest_metric.get("loss", 1.0))
                quality_score = 1.0 / (1.0 + math.exp(loss - 1.0))

        # Build sample comparisons from curated data
        sample_comparisons = []
        curated_path = Path(DATA_DIR) / "distillation" / job_id / "curated.jsonl"
        if curated_path.exists():
            with open(curated_path) as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    try:
                        ex = json.loads(line)
                        sample_comparisons.append({
                            "prompt": ex.get("prompt", "")[:200],
                            "teacher_response": ex.get("response", "")[:500],
                            "student_response": "",
                            "similarity_score": quality_score,
                        })
                    except json.JSONDecodeError:
                        pass

        # Primary GGUF download path
        gguf_path = None
        for qt, path in artifacts.items():
            if qt != "f16":
                gguf_path = path
                break
        if not gguf_path and artifacts:
            gguf_path = next(iter(artifacts.values()))

        results = {
            "quality_score": round(quality_score, 4),
            "cost_savings": 0,
            "speed_improvement": 0,
            "sample_comparisons": sample_comparisons,
            "gguf_download_url": gguf_path,
            "artifacts_count": len(artifacts),
            "quantization_types": list(artifacts.keys()),
        }

        _repo.update_job(tenant_id, job_id, {"results": results})
        print(f"[EXPORT] Wrote results with quality_score={quality_score:.4f}")
    except Exception as e:
        print(f"[EXPORT] Failed to write results: {e}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m lunar_router.distillation.export <config.json>")
        sys.exit(1)
    _run_export(sys.argv[1])
