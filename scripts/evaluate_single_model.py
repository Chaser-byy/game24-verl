#!/usr/bin/env python3
"""Strict single-model evaluation for full-parameter verl checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game24.reward import compute_score
from game24.verifier import verify_solution


WEIGHT_PATTERNS = ("*.safetensors", "model-*.safetensors", "pytorch_model*.bin")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
MIN_MERGE_FREE_BYTES = 4 * 1024**3


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "root_dir": None,
        "outputs_root": "/root/autodl-tmp/outputs",
        "verl_root": "/root/autodl-tmp/verl",
    },
    "model": {
        "source": "latest_run",
        "run_name": None,
        "checkpoint_path": None,
        "hf_model_path": None,
        "step": None,
        "latest_run_patterns": ["game24-grpo-full-param-continue-*", "game24-grpo-full-param-*"],
        "checkpoint": {
            "format": "auto",
            "load_strategy": "auto",
            "auto_merge": True,
            "merged_dir_name": "huggingface_merged",
            "cache_merged_model": True,
            "cleanup_merged_after_evaluation": False,
        },
    },
    "data": {
        "data_dir": "/root/autodl-tmp/game24-verl/data/game24",
        "files": {"val": "val.parquet", "test": "test.parquet"},
        "split": "both",
    },
    "evaluation": {
        "batch_size": 32,
        "max_prompt_length": 192,
        "max_new_tokens": 192,
        "do_sample": False,
        "num_return_sequences": 1,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "device": "cuda",
        "seed": 2026,
        "tokenizer": {
            "padding_side": "left",
            "fix_mistral_regex": True,
            "add_generation_prompt": True,
        },
    },
    "output": {
        "root_dir": "/root/autodl-tmp/outputs/single-model-evaluation",
        "output_dir": None,
        "save_predictions": True,
        "save_prompt_audit": True,
        "save_resolved_config": True,
        "overwrite": False,
    },
}


@dataclass
class Detection:
    checkpoint_format: str
    load_strategy: str
    resolved_model_path: Path | None
    merged_model_path: Path | None
    reason: str
    direct_inference_supported: bool = False
    complete_hf_source: str | None = None
    fsdp_shards: list[str] = field(default_factory=list)


@dataclass
class ResolvedModel:
    source: str
    run_dir: Path | None
    run_name: str
    checkpoint_dir: Path | None
    actor_dir: Path | None
    global_step: int | None
    detection: Detection
    candidate_runs: list[dict[str, Any]] = field(default_factory=list)
    auto_merge_performed: bool = False
    merge_seconds: float = 0.0
    merge_created_this_run: bool = False


@dataclass
class SplitTotals:
    total: int = 0
    exact_correct: int = 0
    format_valid: int = 0
    parse_valid: int = 0
    number_usage_valid: int = 0
    reward_sum: float = 0.0

    def update(self, verification: Any, reward: float) -> None:
        self.total += 1
        self.exact_correct += int(verification.is_correct)
        self.format_valid += int(verification.format_valid)
        self.parse_valid += int(verification.parse_valid)
        self.number_usage_valid += int(verification.numbers_valid)
        self.reward_sum += reward

    def summary(self, *, split: str, elapsed_seconds: float, predictions_path: Path | None) -> dict[str, Any]:
        return {
            "split": split,
            "total": self.total,
            "exact_correct": self.exact_correct,
            "exact_accuracy": self.exact_correct / self.total if self.total else 0.0,
            "format_rate": self.format_valid / self.total if self.total else 0.0,
            "parse_rate": self.parse_valid / self.total if self.total else 0.0,
            "number_usage_rate": self.number_usage_valid / self.total if self.total else 0.0,
            "reward_mean": self.reward_sum / self.total if self.total else 0.0,
            "elapsed_seconds": elapsed_seconds,
            "predictions_path": str(predictions_path) if predictions_path is not None else None,
        }


class TeeStream:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: deep_merge(value, {}) if isinstance(value, Mapping) else value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"config must be a YAML mapping: {config_path}")
    return deep_merge(DEFAULT_CONFIG, loaded)


def project_root_from_config(config: Mapping[str, Any]) -> Path:
    root = config["project"].get("root_dir")
    return Path(root).expanduser().resolve() if root else PROJECT_ROOT


def resolve_path(value: str | None, *, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "model"


def global_step_number(path: Path) -> int | None:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    return int(match.group(1)) if match else None


def has_weight_files(path: Path) -> bool:
    return any(any(path.glob(pattern)) for pattern in WEIGHT_PATTERNS)


def has_tokenizer_files(path: Path) -> bool:
    return any((path / name).exists() for name in TOKENIZER_FILES)


def is_complete_hf_model(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and has_weight_files(path)


def hf_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "config_json": (path / "config.json").is_file(),
        "has_weight_files": has_weight_files(path),
        "has_tokenizer_files": has_tokenizer_files(path),
        "complete": is_complete_hf_model(path),
    }


def checkpoint_actor_dir(checkpoint_dir: Path) -> tuple[Path, Path]:
    if checkpoint_dir.name == "actor":
        return checkpoint_dir.parent, checkpoint_dir
    return checkpoint_dir, checkpoint_dir / "actor"


def detect_checkpoint_format(checkpoint_dir: Path, config: Mapping[str, Any]) -> Detection:
    checkpoint_dir, actor_dir = checkpoint_actor_dir(checkpoint_dir)
    checkpoint_cfg = config["model"]["checkpoint"]
    merged_name = str(checkpoint_cfg["merged_dir_name"])
    preferred_hf_dirs = [actor_dir / merged_name, actor_dir / "huggingface"]

    for hf_dir in preferred_hf_dirs:
        if is_complete_hf_model(hf_dir):
            return Detection(
                checkpoint_format="hf",
                load_strategy="transformers_direct",
                resolved_model_path=hf_dir,
                merged_model_path=actor_dir / merged_name,
                reason=f"found complete Hugging Face model at {hf_dir}",
                complete_hf_source=hf_dir.name,
            )

    shards = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    shard_names = [path.name for path in shards]
    has_fsdp_config = (actor_dir / "fsdp_config.json").is_file()
    has_hf_config = (actor_dir / "huggingface" / "config.json").is_file()
    if shards and has_fsdp_config and has_hf_config:
        return Detection(
            checkpoint_format="verl_fsdp",
            load_strategy="automatic_merge_then_transformers",
            resolved_model_path=None,
            merged_model_path=actor_dir / merged_name,
            reason=(
                "verl v0.7.1 does not expose a simpler stable single-process Transformers inference path "
                "for raw FSDP shards; use official model_merger first"
            ),
            direct_inference_supported=False,
            fsdp_shards=shard_names,
        )

    inspected_hf = [hf_status(path) for path in preferred_hf_dirs if path.exists()]
    missing: list[str] = []
    if shards and not has_fsdp_config:
        missing.append("actor/fsdp_config.json")
    if shards and not has_hf_config:
        missing.append("actor/huggingface/config.json")
    reason = "no complete HF model and no recoverable verl FSDP actor checkpoint"
    if missing:
        reason += f"; missing {', '.join(missing)}"
    if inspected_hf:
        reason += f"; inspected HF dirs: {inspected_hf}"
    return Detection(
        checkpoint_format="unsupported",
        load_strategy="none",
        resolved_model_path=None,
        merged_model_path=actor_dir / merged_name,
        reason=reason,
        fsdp_shards=shard_names,
    )


def validate_detected_checkpoint(checkpoint_dir: Path, config: Mapping[str, Any]) -> Detection:
    detection = detect_checkpoint_format(checkpoint_dir, config)
    if detection.checkpoint_format == "unsupported":
        raise FileNotFoundError(f"checkpoint is not recoverable: {checkpoint_dir}; {detection.reason}")
    return detection


def is_excluded_run_dir(path: Path) -> bool:
    name = path.name.lower()
    excluded_fragments = ("evaluation", "single-model-evaluation", "sft", "lora", "run_metadata")
    return any(fragment in name for fragment in excluded_fragments)


def select_checkpoint_from_run(
    run_dir: Path,
    *,
    step: int | None,
    config: Mapping[str, Any],
    strict_step: bool,
) -> tuple[Path, Detection, list[str]]:
    notes: list[str] = []
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    if step is not None:
        checkpoint_dir = run_dir / f"global_step_{step}"
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"requested global_step_{step} does not exist under {run_dir}")
        detection = validate_detected_checkpoint(checkpoint_dir, config)
        return checkpoint_dir, detection, [f"selected explicit global_step_{step}"]

    candidates: list[tuple[int, Path]] = []
    for child in run_dir.glob("global_step_*"):
        number = global_step_number(child)
        if number is not None and child.is_dir():
            candidates.append((number, child))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no global_step_* directories under {run_dir}")

    for number, checkpoint_dir in candidates:
        detection = detect_checkpoint_format(checkpoint_dir, config)
        if detection.checkpoint_format != "unsupported":
            notes.append(f"selected highest recoverable global_step_{number}: {detection.checkpoint_format}")
            return checkpoint_dir, detection, notes
        notes.append(f"skip global_step_{number}: {detection.reason}")

    message = f"no recoverable checkpoint under {run_dir}; " + " | ".join(notes)
    if strict_step:
        raise FileNotFoundError(message)
    raise FileNotFoundError(message)


def resolve_latest_run(config: Mapping[str, Any], root: Path) -> tuple[Path, Path, Detection, list[dict[str, Any]]]:
    model_cfg = config["model"]
    outputs_root = resolve_path(config["project"]["outputs_root"], base=root)
    assert outputs_root is not None
    patterns = list(model_cfg["latest_run_patterns"])
    candidates: dict[Path, str] = {}
    for pattern in patterns:
        for path in outputs_root.glob(str(pattern)):
            if path.is_dir() and not is_excluded_run_dir(path):
                candidates[path.resolve()] = pattern

    ordered = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    diagnostics: list[dict[str, Any]] = []
    if not ordered:
        raise FileNotFoundError(f"no latest_run candidates found under {outputs_root} with patterns {patterns}")

    requested_step = model_cfg.get("step")
    step = int(requested_step) if requested_step is not None else None
    for path in ordered:
        row: dict[str, Any] = {
            "run_dir": str(path),
            "pattern": candidates[path],
            "mtime": path.stat().st_mtime,
        }
        try:
            checkpoint_dir, detection, notes = select_checkpoint_from_run(
                path, step=step, config=config, strict_step=False
            )
            row.update(
                {
                    "selected": True,
                    "reason": "; ".join(notes),
                    "checkpoint_dir": str(checkpoint_dir),
                    "detected_checkpoint_format": detection.checkpoint_format,
                }
            )
            diagnostics.append(row)
            return path, checkpoint_dir, detection, diagnostics
        except Exception as exc:
            row.update({"selected": False, "reason": str(exc)})
            diagnostics.append(row)

    raise FileNotFoundError(
        "no latest_run candidate has a recoverable checkpoint; candidates="
        + json.dumps(diagnostics, ensure_ascii=False, indent=2)
    )


def resolve_model(config: Mapping[str, Any], root: Path) -> ResolvedModel:
    model_cfg = config["model"]
    source = str(model_cfg["source"])

    if source == "latest_run":
        run_dir, checkpoint_dir, detection, diagnostics = resolve_latest_run(config, root)
        checkpoint_dir, actor_dir = checkpoint_actor_dir(checkpoint_dir)
        step = global_step_number(checkpoint_dir)
        return ResolvedModel(
            source=source,
            run_dir=run_dir,
            run_name=run_dir.name,
            checkpoint_dir=checkpoint_dir,
            actor_dir=actor_dir,
            global_step=step,
            detection=detection,
            candidate_runs=diagnostics,
        )

    if source == "run_name":
        outputs_root = resolve_path(config["project"]["outputs_root"], base=root)
        assert outputs_root is not None
        run_name = model_cfg.get("run_name")
        if not run_name:
            raise ValueError("model.run_name is required when model.source=run_name")
        run_dir = Path(str(run_name)).expanduser()
        if not run_dir.is_absolute():
            run_dir = outputs_root / run_dir
        step_value = model_cfg.get("step")
        step = int(step_value) if step_value is not None else None
        checkpoint_dir, detection, notes = select_checkpoint_from_run(
            run_dir.resolve(), step=step, config=config, strict_step=True
        )
        checkpoint_dir, actor_dir = checkpoint_actor_dir(checkpoint_dir)
        return ResolvedModel(
            source=source,
            run_dir=run_dir.resolve(),
            run_name=run_dir.name,
            checkpoint_dir=checkpoint_dir,
            actor_dir=actor_dir,
            global_step=global_step_number(checkpoint_dir),
            detection=detection,
            candidate_runs=[
                {
                    "run_dir": str(run_dir.resolve()),
                    "selected": True,
                    "reason": "; ".join(notes),
                    "checkpoint_dir": str(checkpoint_dir),
                    "detected_checkpoint_format": detection.checkpoint_format,
                }
            ],
        )

    if source == "checkpoint":
        checkpoint_path = resolve_path(model_cfg.get("checkpoint_path"), base=root)
        if checkpoint_path is None:
            raise ValueError("model.checkpoint_path is required when model.source=checkpoint")
        checkpoint_dir, actor_dir = checkpoint_actor_dir(checkpoint_path.resolve())
        detection = validate_detected_checkpoint(checkpoint_dir, config)
        run_dir = checkpoint_dir.parent if global_step_number(checkpoint_dir) is not None else None
        return ResolvedModel(
            source=source,
            run_dir=run_dir,
            run_name=run_dir.name if run_dir is not None else checkpoint_dir.name,
            checkpoint_dir=checkpoint_dir,
            actor_dir=actor_dir,
            global_step=global_step_number(checkpoint_dir),
            detection=detection,
            candidate_runs=[
                {
                    "run_dir": str(run_dir) if run_dir else None,
                    "selected": True,
                    "reason": "explicit checkpoint_path",
                    "checkpoint_dir": str(checkpoint_dir),
                    "detected_checkpoint_format": detection.checkpoint_format,
                }
            ],
        )

    if source == "direct_hf":
        hf_model_path = resolve_path(model_cfg.get("hf_model_path"), base=root)
        if hf_model_path is None:
            raise ValueError("model.hf_model_path is required when model.source=direct_hf")
        if not is_complete_hf_model(hf_model_path):
            raise FileNotFoundError(f"direct_hf path is not a complete HF model: {hf_status(hf_model_path)}")
        detection = Detection(
            checkpoint_format="hf",
            load_strategy="transformers_direct",
            resolved_model_path=hf_model_path,
            merged_model_path=None,
            reason="direct_hf source points to a complete Hugging Face model",
            complete_hf_source=hf_model_path.name,
        )
        return ResolvedModel(
            source=source,
            run_dir=None,
            run_name=sanitize_name(hf_model_path.name),
            checkpoint_dir=None,
            actor_dir=None,
            global_step=None,
            detection=detection,
            candidate_runs=[],
        )

    raise ValueError(f"unsupported model.source: {source}")


def selected_splits(config: Mapping[str, Any]) -> list[str]:
    split = str(config["data"]["split"]).lower()
    if split == "both":
        return ["val", "test"]
    if split in {"val", "test"}:
        return [split]
    raise ValueError("data.split must be val, test, or both")


def load_problem_file(path: Path) -> list[Any]:
    from scripts.final_evaluation import load_problems

    return load_problems(path)


def resolve_data_files(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    data_dir = resolve_path(config["data"]["data_dir"], base=root)
    assert data_dir is not None
    files = config["data"]["files"]
    result: dict[str, Path] = {}
    for split in selected_splits(config):
        file_name = files.get(split)
        if not file_name:
            raise ValueError(f"data.files.{split} is required")
        path = Path(str(file_name)).expanduser()
        if not path.is_absolute():
            path = data_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"configured {split} parquet does not exist: {path}")
        result[split] = path
    return result


def output_directory(config: Mapping[str, Any], root: Path, resolved: ResolvedModel) -> Path:
    output_cfg = config["output"]
    explicit = resolve_path(output_cfg.get("output_dir"), base=root)
    if explicit is not None:
        return explicit
    output_root = resolve_path(output_cfg["root_dir"], base=root)
    assert output_root is not None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    step_label = f"step_{resolved.global_step}" if resolved.global_step is not None else "step_direct_hf"
    return output_root / sanitize_name(resolved.run_name) / step_label / timestamp


def check_output_dir(path: Path, overwrite: bool, *, dry_run: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory exists and is not empty; set output.overwrite=true: {path}")
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(output_dir: Path) -> Any:
    log_path = output_dir / "evaluation.log"
    handle = log_path.open("a", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, handle)
    sys.stderr = TeeStream(sys.__stderr__, handle)
    print(f"Writing evaluation log to {log_path}")
    return handle


def to_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    return value


def resolved_config_payload(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    root: Path,
    resolved: ResolvedModel,
    data_files: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "project_root": str(root),
        "resolved_run_dir": str(resolved.run_dir) if resolved.run_dir else None,
        "resolved_checkpoint_dir": str(resolved.checkpoint_dir) if resolved.checkpoint_dir else None,
        "resolved_actor_dir": str(resolved.actor_dir) if resolved.actor_dir else None,
        "resolved_global_step": resolved.global_step,
        "detected_checkpoint_format": resolved.detection.checkpoint_format,
        "load_strategy": resolved.detection.load_strategy,
        "direct_fsdp_inference_supported": resolved.detection.direct_inference_supported,
        "resolved_model_path": str(resolved.detection.resolved_model_path)
        if resolved.detection.resolved_model_path
        else None,
        "merged_model_path": str(resolved.detection.merged_model_path)
        if resolved.detection.merged_model_path
        else None,
        "candidate_runs": resolved.candidate_runs,
        "data_files": {name: str(path) for name, path in data_files.items()},
        "output_dir": str(output_dir),
        "config": to_plain(config),
    }


def print_dry_run_summary(payload: Mapping[str, Any]) -> None:
    print("Single-model evaluation dry-run")
    print(f"Resolved run: {payload['resolved_run_dir']}")
    print(f"Resolved checkpoint: {payload['resolved_checkpoint_dir']}")
    print(f"Detected format: {payload['detected_checkpoint_format']}")
    print(f"Load strategy: {payload['load_strategy']}")
    print(f"Merged model target: {payload['merged_model_path']}")
    print(f"Resolved model path: {payload['resolved_model_path']}")
    print(f"Evaluation splits: {', '.join(payload['data_files'].keys())}")
    print("Candidate runs:")
    for row in payload.get("candidate_runs", []):
        print(f"  - selected={row.get('selected')} run={row.get('run_dir')} reason={row.get('reason')}")
    print("Resolved config:")
    print(yaml.safe_dump(to_plain(payload), sort_keys=False, allow_unicode=True))


def existing_parent(path: Path) -> Path:
    current = path if path.is_dir() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def check_merge_disk_space(target_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    parent = existing_parent(target_dir)
    usage = shutil.disk_usage(parent)
    result = {
        "path_checked": str(parent),
        "free_bytes": usage.free,
        "required_free_bytes": MIN_MERGE_FREE_BYTES,
        "checkpoint_path": str(checkpoint_dir),
    }
    if usage.free < MIN_MERGE_FREE_BYTES:
        raise RuntimeError(
            "not enough free disk space for automatic merge: "
            f"free={usage.free / 1024**3:.2f} GiB, "
            f"required={MIN_MERGE_FREE_BYTES / 1024**3:.2f} GiB, checkpoint={checkpoint_dir}"
        )
    return result


def safe_remove_incomplete_merged_dir(target_dir: Path, actor_dir: Path, merged_dir_name: str) -> None:
    if target_dir.name != merged_dir_name or target_dir.parent.resolve() != actor_dir.resolve():
        raise RuntimeError(f"refusing to remove unexpected merge target: {target_dir}")
    if target_dir.exists() and not is_complete_hf_model(target_dir):
        print(f"Removing incomplete cached merge directory before rebuilding: {target_dir}")
        shutil.rmtree(target_dir)


def merge_env(config: Mapping[str, Any], root: Path) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(root)]
    verl_root = resolve_path(config["project"].get("verl_root"), base=root)
    if verl_root is not None:
        paths.append(str(verl_root))
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_automatic_merge(resolved: ResolvedModel, config: Mapping[str, Any], root: Path) -> None:
    if resolved.detection.checkpoint_format != "verl_fsdp":
        return
    checkpoint_cfg = config["model"]["checkpoint"]
    if not checkpoint_cfg.get("auto_merge", True):
        raise RuntimeError(f"checkpoint requires merge but model.checkpoint.auto_merge=false: {resolved.checkpoint_dir}")
    assert resolved.actor_dir is not None
    assert resolved.checkpoint_dir is not None
    assert resolved.detection.merged_model_path is not None
    target_dir = resolved.detection.merged_model_path

    if is_complete_hf_model(target_dir):
        print(f"Reusing cached merged Hugging Face model: {target_dir}")
        resolved.detection.resolved_model_path = target_dir
        resolved.detection.checkpoint_format = "hf"
        resolved.detection.load_strategy = "transformers_direct"
        return

    disk = check_merge_disk_space(target_dir, resolved.checkpoint_dir)
    print("Automatic merge is required")
    print(f"  checkpoint_path={resolved.checkpoint_dir}")
    print(f"  checkpoint_format=verl_fsdp")
    print(f"  direct_load_reason={resolved.detection.reason}")
    print(f"  merge_target={target_dir}")
    print(
        "  disk_free="
        f"{disk['free_bytes'] / 1024**3:.2f} GiB "
        f"required={disk['required_free_bytes'] / 1024**3:.2f} GiB"
    )

    safe_remove_incomplete_merged_dir(target_dir, resolved.actor_dir, str(checkpoint_cfg["merged_dir_name"]))
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(resolved.actor_dir),
        "--target_dir",
        str(target_dir),
    ]
    print("Running merge command:")
    print("  " + " ".join(cmd))

    start = time.monotonic()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=merge_env(config, root),
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    resolved.merge_seconds = time.monotonic() - start
    if return_code != 0:
        raise RuntimeError(f"verl model_merger failed with exit code {return_code}")

    if not is_complete_hf_model(target_dir):
        raise RuntimeError(f"merge finished but target is not a complete HF model: {hf_status(target_dir)}")
    if not has_tokenizer_files(target_dir):
        raise RuntimeError(f"merge finished but tokenizer files are not available: {target_dir}")

    resolved.auto_merge_performed = True
    resolved.merge_created_this_run = True
    resolved.detection.resolved_model_path = target_dir
    resolved.detection.checkpoint_format = "hf"
    resolved.detection.load_strategy = "automatic_merge_then_transformers"
    print(f"Automatic merge completed in {resolved.merge_seconds:.1f}s: {target_dir}")


def torch_dtype(dtype_name: str) -> Any:
    import torch

    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported evaluation.dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(model_path: Path, config: Mapping[str, Any]) -> tuple[Any, Any, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    eval_cfg = config["evaluation"]
    tokenizer_cfg = eval_cfg["tokenizer"]
    device = str(eval_cfg["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("evaluation.device=cuda but CUDA is not available")

    set_seed(int(eval_cfg["seed"]))
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.padding_side = str(tokenizer_cfg["padding_side"])
    if bool(tokenizer_cfg.get("fix_mistral_regex", False)):
        setattr(tokenizer, "fix_mistral_regex", True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(str(eval_cfg["dtype"])),
        "trust_remote_code": True,
    }
    attention = eval_cfg.get("attention_implementation")
    if attention:
        model_kwargs["attn_implementation"] = str(attention)
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    return model, tokenizer, time.monotonic() - start


def render_prompt_and_ids(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    add_generation_prompt: bool,
) -> tuple[str, list[int]]:
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return str(rendered), [int(token_id) for token_id in token_ids]

    rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    if add_generation_prompt:
        rendered += "\nassistant:"
    tokenized = tokenizer(rendered, add_special_tokens=False)
    return rendered, [int(token_id) for token_id in tokenized["input_ids"]]


def prompt_text(problem: Problem, tokenizer: Any, config: Mapping[str, Any]) -> str:
    rendered, token_ids = render_prompt_and_ids(
        problem.prompt_messages,
        tokenizer,
        add_generation_prompt=bool(config["evaluation"]["tokenizer"]["add_generation_prompt"]),
    )
    max_prompt_length = int(config["evaluation"]["max_prompt_length"])
    if len(token_ids) > max_prompt_length:
        raise ValueError(
            f"prompt for problem {problem.problem_id} has {len(token_ids)} tokens, "
            f"exceeding evaluation.max_prompt_length={max_prompt_length}"
        )
    return rendered


def generate_batch(model: Any, tokenizer: Any, problems: Sequence[Problem], config: Mapping[str, Any]) -> list[str]:
    import torch

    eval_cfg = config["evaluation"]
    prompts = [prompt_text(problem, tokenizer, config) for problem in problems]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(eval_cfg["max_new_tokens"]),
        "do_sample": bool(eval_cfg["do_sample"]),
        "num_return_sequences": int(eval_cfg["num_return_sequences"]),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    prompt_length = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)


def ground_truth(problem: Problem) -> dict[str, Any]:
    return {"numbers": problem.numbers, "target": problem.target, "solvable": problem.solvable}


def canonical_id(numbers: Sequence[int]) -> list[int]:
    return sorted(int(number) for number in numbers)


def evaluate_split(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    split: str,
    problems: Sequence[Problem],
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    eval_cfg = config["evaluation"]
    batch_size = int(eval_cfg["batch_size"])
    predictions_path = output_dir / "predictions" / f"{split}.jsonl"
    if config["output"].get("save_predictions", True):
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        handle = predictions_path.open("w", encoding="utf-8")
    else:
        handle = None
        predictions_path = None  # type: ignore[assignment]

    totals = SplitTotals()
    start_time = time.monotonic()
    total_batches = math.ceil(len(problems) / batch_size) if problems else 0
    try:
        for batch_index, start in enumerate(range(0, len(problems), batch_size), start=1):
            batch = problems[start : start + batch_size]
            responses = generate_batch(model, tokenizer, batch, config)
            if len(responses) != len(batch):
                raise RuntimeError(f"expected {len(batch)} responses, got {len(responses)}")

            for offset, (problem, response) in enumerate(zip(batch, responses, strict=True)):
                verification = verify_solution(response, problem.numbers, target=problem.target)
                reward = float(compute_score("game24", response, ground_truth(problem)))
                totals.update(verification, reward)
                record = {
                    "model_name": model_name,
                    "split": split,
                    "index": start + offset,
                    "problem_id": problem.problem_id,
                    "numbers": problem.numbers,
                    "canonical_id": canonical_id(problem.numbers),
                    "target": problem.target,
                    "solvable": problem.solvable,
                    "generation_text": response,
                    "answer": verification.expression,
                    "is_correct": verification.is_correct,
                    "format_valid": verification.format_valid,
                    "parse_valid": verification.parse_valid,
                    "number_usage_valid": verification.numbers_valid,
                    "error_reason": verification.error_reason,
                    "reward": reward,
                    "verification": verification.to_dict(),
                }
                if handle is not None:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if handle is not None:
                handle.flush()
            elapsed = time.monotonic() - start_time
            accuracy = totals.exact_correct / totals.total if totals.total else 0.0
            print(
                f"[{model_name}][{split}] batch {batch_index}/{total_batches} "
                f"completed={min(start + len(batch), len(problems))}/{len(problems)} "
                f"strict_correct={totals.exact_correct}/{totals.total} "
                f"exact_accuracy={accuracy:.6f} elapsed={elapsed:.1f}s"
            )
    finally:
        if handle is not None:
            handle.close()

    return totals.summary(
        split=split,
        elapsed_seconds=time.monotonic() - start_time,
        predictions_path=predictions_path,
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_plain(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_prompt_audit(
    *,
    path: Path,
    first_problem: Problem,
    tokenizer: Any,
    config_path: Path,
    config: Mapping[str, Any],
    resolved: ResolvedModel,
) -> dict[str, Any]:
    rendered, token_ids = render_prompt_and_ids(
        first_problem.prompt_messages,
        tokenizer,
        add_generation_prompt=bool(config["evaluation"]["tokenizer"]["add_generation_prompt"]),
    )
    audit = {
        "raw_messages": first_problem.prompt_messages,
        "rendered_prompt": rendered,
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "numbers": first_problem.numbers,
        "target": first_problem.target,
        "checkpoint_path": str(resolved.checkpoint_dir) if resolved.checkpoint_dir else None,
        "final_model_path": str(resolved.detection.resolved_model_path)
        if resolved.detection.resolved_model_path
        else None,
        "checkpoint_format": resolved.detection.checkpoint_format,
        "load_strategy": resolved.detection.load_strategy,
        "auto_merge_performed": resolved.auto_merge_performed,
        "config_path": str(config_path),
        "add_generation_prompt": bool(config["evaluation"]["tokenizer"]["add_generation_prompt"]),
        "tokenizer_padding_side": getattr(tokenizer, "padding_side", None),
        "tokenizer_pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "tokenizer_eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "format_requirement": "<think>...</think><answer>...</answer>",
    }
    write_json(path, audit)
    return audit


def validate_evaluation_config(config: Mapping[str, Any]) -> None:
    eval_cfg = config["evaluation"]
    for key in ("batch_size", "max_prompt_length", "max_new_tokens", "num_return_sequences"):
        if int(eval_cfg[key]) <= 0:
            raise ValueError(f"evaluation.{key} must be positive")
    if bool(eval_cfg["do_sample"]):
        raise ValueError("single-model strict evaluation is greedy Pass@1; evaluation.do_sample must be false")
    if int(eval_cfg["num_return_sequences"]) != 1:
        raise ValueError(
            "single-model strict evaluation is greedy Pass@1; evaluation.num_return_sequences must be 1"
        )


def cleanup_if_requested(resolved: ResolvedModel, config: Mapping[str, Any]) -> None:
    checkpoint_cfg = config["model"]["checkpoint"]
    cleanup_requested = bool(checkpoint_cfg.get("cleanup_merged_after_evaluation", False)) or not bool(
        checkpoint_cfg.get("cache_merged_model", True)
    )
    if not cleanup_requested:
        return
    if not resolved.merge_created_this_run:
        print("Merged-model cleanup requested, but no merged model was created in this run; nothing removed.")
        return
    target = resolved.detection.resolved_model_path
    if target is None or resolved.actor_dir is None:
        return
    safe_remove_incomplete_merged_dir(target, resolved.actor_dir, str(checkpoint_cfg["merged_dir_name"]))
    if target.exists() and target.name == str(checkpoint_cfg["merged_dir_name"]) and target.parent == resolved.actor_dir:
        print(f"Removing merged model created by this run: {target}")
        shutil.rmtree(target)


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    root = project_root_from_config(config)
    validate_evaluation_config(config)
    resolved = resolve_model(config, root)
    data_files = resolve_data_files(config, root)
    output_dir = output_directory(config, root, resolved)
    check_output_dir(output_dir, bool(config["output"]["overwrite"]), dry_run=args.dry_run)
    payload = resolved_config_payload(
        config=config,
        config_path=config_path,
        root=root,
        resolved=resolved,
        data_files=data_files,
        output_dir=output_dir,
    )

    if args.dry_run:
        if resolved.detection.checkpoint_format == "verl_fsdp" and resolved.detection.merged_model_path:
            check_merge_disk_space(resolved.detection.merged_model_path, resolved.checkpoint_dir or output_dir)
        print_dry_run_summary(payload)
        return

    log_handle = setup_logging(output_dir)
    try:
        print("Single-model strict evaluation")
        print(yaml.safe_dump(to_plain(payload), sort_keys=False, allow_unicode=True))
        if config["output"].get("save_resolved_config", True):
            write_yaml(output_dir / "resolved_config.yaml", payload)

        run_automatic_merge(resolved, config, root)
        if resolved.detection.resolved_model_path is None:
            raise RuntimeError(f"could not resolve a loadable model path: {resolved.detection}")

        model_name = (
            f"{sanitize_name(resolved.run_name)}_step_{resolved.global_step}"
            if resolved.global_step is not None
            else sanitize_name(resolved.run_name)
        )
        model, tokenizer, model_load_seconds = load_model_and_tokenizer(resolved.detection.resolved_model_path, config)
        summaries: dict[str, Any] = {}
        try:
            datasets = {split: load_problem_file(path) for split, path in data_files.items()}
            first_split = next(iter(datasets))
            if config["output"].get("save_prompt_audit", True):
                write_prompt_audit(
                    path=output_dir / "prompt_audit.json",
                    first_problem=datasets[first_split][0],
                    tokenizer=tokenizer,
                    config_path=config_path,
                    config=config,
                    resolved=resolved,
                )
            for split, problems in datasets.items():
                summary = evaluate_split(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    split=split,
                    problems=problems,
                    output_dir=output_dir,
                    config=config,
                )
                summaries[split] = summary
                write_json(output_dir / f"{split}_results.json", summary)
        finally:
            del model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        results = {
            "resolved_run_dir": str(resolved.run_dir) if resolved.run_dir else None,
            "resolved_checkpoint_dir": str(resolved.checkpoint_dir) if resolved.checkpoint_dir else None,
            "detected_checkpoint_format": resolved.detection.checkpoint_format,
            "load_strategy": resolved.detection.load_strategy,
            "direct_fsdp_inference_supported": resolved.detection.direct_inference_supported,
            "auto_merge_performed": resolved.auto_merge_performed,
            "merged_model_path": str(resolved.detection.merged_model_path)
            if resolved.detection.merged_model_path
            else None,
            "resolved_model_path": str(resolved.detection.resolved_model_path),
            "resolved_global_step": resolved.global_step,
            "model_name": model_name,
            "model_load_seconds": model_load_seconds,
            "merge_seconds": resolved.merge_seconds,
            "evaluation_config": to_plain(config["evaluation"]),
            "data_files": {name: str(path) for name, path in data_files.items()},
            "candidate_runs": resolved.candidate_runs,
            "strict_exact_accuracy_definition": (
                "verification.is_correct: exactly one answer tag, AST-whitelisted expression parses, "
                "input number multiset matches exactly, and Fraction value equals target"
            ),
            "summaries": summaries,
        }
        if "val" in summaries:
            results["val"] = summaries["val"]
        if "test" in summaries:
            results["test"] = summaries["test"]
        write_json(output_dir / "results.json", results)
        cleanup_if_requested(resolved, config)
        print("Evaluation complete")
        print(json.dumps(to_plain(results), indent=2, sort_keys=True, ensure_ascii=False))
    finally:
        log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict single-model Game24 evaluation.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/evaluation/single_model/default.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    args = parser.parse_args()
    if args.print_resolved_config:
        args.dry_run = True
    return args


def main() -> None:
    try:
        run(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
