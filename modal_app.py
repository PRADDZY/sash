"""Modal entry point for pinned downloads, evaluation, and analysis.

Examples:
    modal run modal_app.py --stage download
    modal run modal_app.py --stage prepare
    modal run modal_app.py --stage pilot
    modal run modal_app.py --stage infer --model base
    modal run modal_app.py --stage infer --model finetuned
    modal run modal_app.py --stage analyze
    modal run modal_app.py --stage replication-pilot
    modal run modal_app.py --stage infer --model smolvlm
    modal run modal_app.py --stage infer --model llava_onevision
    modal run modal_app.py --stage analyze-replication
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP_NAME = "sash-vlm-safety-audit"
MODEL_VOLUME_NAME = "sash-vlm-safety-models"
ARTIFACT_VOLUME_NAME = "sash-vlm-safety"
MODEL_VOLUME_PATH = Path("/models")
ARTIFACT_VOLUME_PATH = Path("/vol")
MANIFEST_PATH = ARTIFACT_VOLUME_PATH / "data" / "manifest.jsonl"
PILOT_MANIFEST_PATH = ARTIFACT_VOLUME_PATH / "data" / "pilot_manifest.jsonl"
PREDICTION_DIR = ARTIFACT_VOLUME_PATH / "predictions"
MODEL_PATHS = {
    "base": MODEL_VOLUME_PATH / "base",
    "finetuned": MODEL_VOLUME_PATH / "finetuned",
    "phi4": MODEL_VOLUME_PATH / "phi4",
    "smolvlm": MODEL_VOLUME_PATH / "smolvlm",
    "llava_onevision": MODEL_VOLUME_PATH / "llava_onevision",
}
MODEL_REPOSITORIES = {
    "base": (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    ),
    "finetuned": (
        "praddzy/Qwen-VizWiz-8B",
        "984042ce162749db644dd3e6be19d6f20aafe3cf",
    ),
    "phi4": (
        "microsoft/Phi-4-multimodal-instruct",
        "93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
    ),
    "smolvlm": (
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "482adb537c021c86670beed01cd58990d01e72e4",
    ),
    "llava_onevision": (
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "0d50680527681998e456c7b78950205bedd8a068",
    ),
}
PRIMARY_MODEL_KEYS = ("base", "finetuned", "smolvlm", "llava_onevision")
REPLICATION_MODEL_KEYS = ("smolvlm", "llava_onevision")
DATASET_ID = "lmms-lab/VizWiz-VQA"
DATASET_REVISION = "d428a2dae984f79cf1b9d99467dfa883e0c30686"
NATURAL_ROWS = 4301
SHIFT_ROWS = 1200
GPU_TYPE = "A10"
GPU_HOURLY_USD = 1.1016
PRIMARY_GPU_BUDGET_USD = 55.0
REPLICATION_GPU_BUDGET_USD = 20.0
WANDB_MODAL_SECRET_NAME = os.getenv("SASH_MODAL_WANDB_SECRET_NAME", "").strip()
WANDB_SECRETS = (
    [modal.Secret.from_name(WANDB_MODAL_SECRET_NAME)] if WANDB_MODAL_SECRET_NAME else []
)

app = modal.App(APP_NAME, tags={"project": "sash-vlm-safety"})
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)

download_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface_hub==0.36.0",
)

data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "datasets==4.8.3",
        "huggingface_hub==0.36.0",
        "Pillow==12.1.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("src/sash_audit", remote_path="/root/sash_audit", copy=True)
)

runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "accelerate==1.12.0",
        "backoff==2.2.1",
        "matplotlib==3.10.7",
        "numpy==2.2.6",
        "num2words==0.5.14",
        "pandas==2.3.3",
        "peft==0.20.0",
        "Pillow==12.1.0",
        "scipy==1.16.3",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.57.6",
        "wandb==0.28.1",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/sash_audit", remote_path="/root/sash_audit", copy=True)
)


@app.function(
    image=download_image,
    cpu=4,
    memory=8192,
    timeout=6 * 60 * 60,
    volumes={str(MODEL_VOLUME_PATH): model_volume},
)
def download_models() -> dict[str, str]:
    """Download both pinned checkpoints without paying for a GPU."""

    from huggingface_hub import snapshot_download

    downloaded: dict[str, str] = {}
    for model_key in PRIMARY_MODEL_KEYS:
        repository, revision = MODEL_REPOSITORIES[model_key]
        local_path = MODEL_PATHS[model_key]
        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=local_path,
        )
        downloaded[model_key] = str(local_path)
    model_volume.commit()
    return downloaded


@app.function(
    image=download_image,
    cpu=4,
    memory=8192,
    timeout=6 * 60 * 60,
    volumes={str(MODEL_VOLUME_PATH): model_volume},
)
def download_fallback_model() -> dict[str, str]:
    """Explicitly refresh the predeclared LLaVA fallback checkpoint."""

    from huggingface_hub import snapshot_download

    repository, revision = MODEL_REPOSITORIES["llava_onevision"]
    local_path = MODEL_PATHS["llava_onevision"]
    snapshot_download(repo_id=repository, revision=revision, local_dir=local_path)
    model_volume.commit()
    return {"llava_onevision": str(local_path)}


@app.function(
    image=data_image,
    cpu=4,
    memory=12_288,
    timeout=8 * 60 * 60,
    volumes={str(ARTIFACT_VOLUME_PATH): artifact_volume},
)
def prepare_dataset(seed: int = 20260712) -> dict[str, int | str]:
    from datasets import load_dataset

    from sash_audit.data import (
        build_manifest,
        read_manifest,
        validate_manifest,
        validate_pilot_manifest,
        write_manifest,
    )

    artifact_volume.reload()
    if MANIFEST_PATH.exists() and PILOT_MANIFEST_PATH.exists():
        rows = read_manifest(MANIFEST_PATH)
        pilot_rows = read_manifest(PILOT_MANIFEST_PATH)
        try:
            validate_manifest(rows)
            validate_pilot_manifest(pilot_rows)
        except ValueError:
            pass
        else:
            return {
                "status": "already_prepared",
                "manifest_rows": len(rows),
                "natural_rows": sum(row.regime == "natural" for row in rows),
                "shift_rows": sum(row.regime == "shift" for row in rows),
            }

    dataset = load_dataset(
        DATASET_ID,
        split="val",
        revision=DATASET_REVISION,
        streaming=True,
    )
    rows, pilot_rows = build_manifest(
        dataset,
        ARTIFACT_VOLUME_PATH / "data",
        seed=seed,
        expected_source_rows=4319,
    )
    validate_manifest(rows)
    validate_pilot_manifest(pilot_rows)
    write_manifest(rows, MANIFEST_PATH)
    write_manifest(pilot_rows, PILOT_MANIFEST_PATH)
    artifact_volume.commit()
    return {
        "status": "prepared",
        "manifest_rows": len(rows),
        "natural_rows": sum(row.regime == "natural" for row in rows),
        "shift_rows": sum(row.regime == "shift" for row in rows),
    }


def _project_cost(stats: dict[str, float | int]) -> float:
    natural_written = int(stats["natural_written"])
    shift_written = int(stats["shift_written"])
    if not natural_written or not shift_written:
        raise RuntimeError("pilot must contain both natural and shifted cases")
    natural_rate = float(stats["natural_seconds"]) / natural_written
    shift_rate = float(stats["shift_seconds"]) / shift_written
    projected_seconds = (
        float(stats["load_seconds"])
        + natural_rate * NATURAL_ROWS
        + shift_rate * SHIFT_ROWS
    )
    return projected_seconds / 3600 * GPU_HOURLY_USD


@app.function(
    image=runtime_image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=32_768,
    timeout=18 * 60 * 60,
    retries=1,
    max_containers=1,
    volumes={
        str(MODEL_VOLUME_PATH): model_volume.read_only(),
        str(ARTIFACT_VOLUME_PATH): artifact_volume,
    },
    secrets=WANDB_SECRETS,
)
def infer_model(model_key: str, pilot: bool = False) -> dict[str, float | int | str]:
    from sash_audit.data import read_manifest
    from sash_audit.inference import run_inference

    if model_key not in MODEL_PATHS:
        raise ValueError(f"unknown model key: {model_key}")
    artifact_volume.reload()
    manifest_path = PILOT_MANIFEST_PATH if pilot else MANIFEST_PATH
    if not manifest_path.exists():
        raise RuntimeError("dataset manifest is missing; run --stage prepare")
    if not (MODEL_PATHS[model_key] / "config.json").exists():
        raise RuntimeError("model weights are missing; run --stage download")
    selected = read_manifest(manifest_path)
    filename = f"pilot_{model_key}.jsonl" if pilot else f"{model_key}.jsonl"
    stats = run_inference(
        selected,
        model_key=model_key,
        model_path=MODEL_PATHS[model_key],
        output_path=PREDICTION_DIR / filename,
        commit_callback=artifact_volume.commit,
    )
    if pilot and int(stats["written"]):
        (PREDICTION_DIR / f"pilot_{model_key}_stats.json").write_text(
            json.dumps(stats, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_volume.commit()
    result: dict[str, float | int | str] = {"model_key": model_key, **stats}
    if pilot and int(stats["written"]):
        result["projected_gpu_cost_usd"] = _project_cost(stats)
    return result


@app.function(
    image=runtime_image,
    cpu=4,
    memory=16_384,
    timeout=3 * 60 * 60,
    volumes={str(ARTIFACT_VOLUME_PATH): artifact_volume},
)
def analyze() -> dict[str, str | int]:
    import csv
    import shutil

    from sash_audit.analysis import read_predictions, write_analysis

    artifact_volume.reload()
    paths = [PREDICTION_DIR / "base.jsonl", PREDICTION_DIR / "finetuned.jsonl"]
    frame = read_predictions(paths)
    output_dir = ARTIFACT_VOLUME_PATH / "analysis"
    write_analysis(frame, output_dir)
    image_dir = output_dir / "failure_images"
    image_dir.mkdir(exist_ok=True)
    with (output_dir / "failure_audit.csv").open(encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))
    image_paths = {
        row[column]
        for row in audit_rows
        for column in ("image_path", "clear_image_path")
        if row[column]
    }
    for image_path in image_paths:
        source = Path(image_path)
        shutil.copy2(source, image_dir / source.name)
    bundle = shutil.make_archive(str(ARTIFACT_VOLUME_PATH / "analysis_bundle"), "zip", output_dir)
    artifact_volume.commit()
    return {
        "rows": len(frame),
        "metrics": str(output_dir / "metrics.csv"),
        "paired_differences": str(output_dir / "paired_differences.csv"),
        "paper_results": str(output_dir / "paper_results.tex"),
        "failure_images": str(image_dir),
        "bundle": bundle,
    }


@app.function(
    image=runtime_image,
    cpu=4,
    memory=16_384,
    timeout=3 * 60 * 60,
    volumes={str(ARTIFACT_VOLUME_PATH): artifact_volume},
)
def analyze_replication() -> dict[str, str | int]:
    from sash_audit.analysis import read_predictions
    from sash_audit.replication import write_replication_analysis

    artifact_volume.reload()
    paths = [PREDICTION_DIR / f"{key}.jsonl" for key in REPLICATION_MODEL_KEYS]
    frame = read_predictions(paths)
    output_dir = ARTIFACT_VOLUME_PATH / "analysis_replication"
    outputs = write_replication_analysis(frame, output_dir, strict_counts=True)
    artifact_volume.commit()
    return {"rows": len(frame), **outputs}


@app.local_entrypoint()
def main(stage: str, model: str = "") -> None:
    if stage == "download":
        print(download_models.remote())
    elif stage == "download-fallback":
        print(download_fallback_model.remote())
    elif stage == "prepare":
        print(prepare_dataset.remote())
    elif stage == "pilot":
        results = [
            infer_model.remote(key, pilot=True)
            for key in ("base", "finetuned")
        ]
        costs = [
            float(result["projected_gpu_cost_usd"])
            for result in results
            if "projected_gpu_cost_usd" in result
        ]
        if len(costs) != 2:
            print({"models": results, "status": "pilot rows already checkpointed"})
            return
        projected = sum(costs)
        print({"models": results, "projected_primary_gpu_cost_usd": projected})
        if projected > PRIMARY_GPU_BUDGET_USD:
            raise RuntimeError(
                f"projected GPU cost {projected:.2f} USD exceeds "
                f"the {PRIMARY_GPU_BUDGET_USD:.0f} USD gate"
            )
    elif stage == "replication-pilot":
        results = [infer_model.remote(key, pilot=True) for key in REPLICATION_MODEL_KEYS]
        costs = [
            float(result["projected_gpu_cost_usd"])
            for result in results
            if "projected_gpu_cost_usd" in result
        ]
        if len(costs) != 2:
            print({"models": results, "status": "pilot rows already checkpointed"})
            return
        projected = sum(costs)
        print({"models": results, "projected_replication_gpu_cost_usd": projected})
        if projected > REPLICATION_GPU_BUDGET_USD:
            raise RuntimeError(
                f"projected replication GPU cost {projected:.2f} USD exceeds "
                f"the {REPLICATION_GPU_BUDGET_USD:.0f} USD gate"
            )
    elif stage == "infer":
        if model not in MODEL_PATHS:
            raise ValueError(f"--model must be one of: {', '.join(MODEL_PATHS)}")
        print(infer_model.remote(model, pilot=False))
    elif stage == "analyze":
        print(analyze.remote())
    elif stage == "analyze-replication":
        print(analyze_replication.remote())
    else:
        raise ValueError(
            "--stage must be download, download-fallback, prepare, pilot, "
            "replication-pilot, infer, analyze, or analyze-replication"
        )
