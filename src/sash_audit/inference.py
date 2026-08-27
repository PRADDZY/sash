"""Pinned Qwen3-VL inference and visual-sensitivity scoring."""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .data import ManifestRow
from .metrics import canonical_prediction, is_false_answer_on_unanswerable, vqa_accuracy
from .tracking import create_tracker

SYSTEM_PROMPT = (
    "Answer the visual question using only evidence visible in the image. "
    "If there is not enough visual evidence, answer exactly 'unanswerable'. "
    "Return only a short answer, without explanation."
)
MODEL_SPECS = {
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
REPLICATION_MODEL_KEYS = ("smolvlm", "llava_onevision")
FALLBACK_MODEL_KEY = "llava_onevision"
QWEN_MODEL_KEYS = ("base", "finetuned")
MIN_PIXELS = 64 * 32 * 32
MAX_PIXELS = 1024 * 32 * 32
LLAVA_MAX_PIXELS = 512 * 32 * 32
EMPTY_SCORE = -100.0
QWEN_GENERATION_SCORE_TOLERANCE = 0.05
GENERIC_GENERATION_SCORE_TOLERANCE = 0.10


@dataclass(frozen=True)
class Prediction:
    case_id: str
    row_index: int
    model_key: str
    model_id: str
    model_revision: str
    evaluation_split: str
    regime: str
    corruption: str
    source_case_id: str | None
    image_path: str
    clear_image_path: str | None
    question: str
    reference_answers: tuple[str, ...]
    answerable: bool
    answer_type: str
    generated_answer: str
    normalized_answer: str
    generated_token_ids: tuple[int, ...]
    real_token_logprobs: tuple[float, ...]
    blank_token_logprobs: tuple[float, ...]
    clear_token_logprobs: tuple[float, ...] | None
    vqa_score: float
    entirely_wrong: bool
    false_answer_on_unanswerable: bool
    mean_logprob: float
    blank_logprob: float
    grounding_score: float
    clear_logprob: float | None
    clear_grounding_score: float | None
    acquisition_score: float
    generated_tokens: int
    image_grid_thw: tuple[int, ...]


def _conversation(image: Image.Image, question: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]


def _finite_mean(values: list[float] | tuple[float, ...]) -> float:
    return float(sum(values) / len(values)) if values else EMPTY_SCORE


class QwenVLAuditor:
    """Single-checkpoint adapter with exact generated-token teacher forcing."""

    def __init__(self, model_path: str | Path, *, max_new_tokens: int = 32) -> None:
        import torch
        from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            str(model_path),
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            fix_mistral_regex=True,
            local_files_only=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        self.device = next(self.model.parameters()).device
        tokenizer = self.processor.tokenizer
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        self.generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_k=None,
            top_p=None,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_id,
        )
        self.model.generation_config = self.generation_config
        torch.cuda.reset_peak_memory_stats()

    @property
    def peak_memory_gb(self) -> float:
        return float(self.torch.cuda.max_memory_reserved() / 1024**3)

    def close(self) -> None:
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()

    def _processor_inputs(self, image: Image.Image, question: str) -> dict[str, Any]:
        conversation = _conversation(image, question)
        encoded = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

    def _score_tokens(self, prompt_inputs: dict[str, Any], answer_ids: Any) -> list[float]:
        if answer_ids.numel() == 0:
            return []
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        model_inputs = dict(prompt_inputs)
        model_inputs["input_ids"] = self.torch.cat(
            [prompt_inputs["input_ids"], answer_ids.unsqueeze(0)], dim=1
        )
        model_inputs["attention_mask"] = self.torch.cat(
            [
                prompt_inputs["attention_mask"],
                self.torch.ones(
                    (1, answer_ids.shape[0]),
                    dtype=prompt_inputs["attention_mask"].dtype,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        positions = self.torch.arange(
            prompt_length - 1,
            prompt_length + answer_ids.shape[0] - 1,
            device=self.device,
        )
        with self.torch.inference_mode():
            output = self.model(
                **model_inputs,
                use_cache=False,
                logits_to_keep=positions,
                return_dict=True,
            )
        logits = output.logits
        if logits.shape[1] != answer_ids.shape[0]:
            # Older patch releases may ignore tensor-valued logits_to_keep.
            logits = logits[:, prompt_length - 1 : prompt_length + answer_ids.shape[0] - 1]
        values = (
            logits.double()
            .log_softmax(dim=-1)
            .gather(-1, answer_ids.view(1, -1, 1))
            .squeeze(0)
            .squeeze(-1)
        )
        return [float(value) for value in values.cpu().tolist()]

    def audit(
        self,
        image: Image.Image,
        question: str,
        *,
        clear_image: Image.Image | None = None,
        verify_generation_score: bool = False,
    ) -> dict[str, object]:
        """Generate once, then score those exact tokens under counterfactual images."""

        real_inputs = self._processor_inputs(image, question)
        prompt_length = int(real_inputs["input_ids"].shape[1])
        with self.torch.inference_mode():
            output = self.model.generate(
                **real_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = output.sequences[0, prompt_length : prompt_length + len(output.scores)]
        selected_logprobs = [
            float(step_scores[0].double().log_softmax(dim=-1)[generated_ids[index]])
            for index, step_scores in enumerate(output.scores)
        ]
        special_ids = set(self.processor.tokenizer.all_special_ids)
        usable_positions = [
            index
            for index, token_id in enumerate(generated_ids.tolist())
            if token_id not in special_ids
        ]
        answer_ids = generated_ids[usable_positions]
        real_logprobs = [selected_logprobs[index] for index in usable_positions]
        answer = self.processor.decode(answer_ids, skip_special_tokens=True).strip()

        blank = Image.new("RGB", image.size, color=(127, 127, 127))
        try:
            blank_inputs = self._processor_inputs(blank, question)
        finally:
            blank.close()
        if not self.torch.equal(real_inputs["input_ids"], blank_inputs["input_ids"]):
            raise RuntimeError("real and blank prompt token IDs differ")
        if not self.torch.equal(real_inputs["image_grid_thw"], blank_inputs["image_grid_thw"]):
            raise RuntimeError("real and blank visual grids differ")
        blank_logprobs = self._score_tokens(blank_inputs, answer_ids)
        if len(blank_logprobs) != len(real_logprobs):
            raise RuntimeError("blank and real answer-token counts differ")

        if verify_generation_score and answer_ids.numel():
            forced_real = self._score_tokens(real_inputs, answer_ids)
            max_difference = max(
                abs(generated - forced)
                for generated, forced in zip(real_logprobs, forced_real, strict=True)
            )
            if max_difference > QWEN_GENERATION_SCORE_TOLERANCE:
                raise RuntimeError(
                    f"generation/teacher-forcing log-prob mismatch: {max_difference:.4f}"
                )

        clear_logprobs: list[float] | None = None
        if clear_image is not None:
            clear_inputs = self._processor_inputs(clear_image, question)
            if not self.torch.equal(real_inputs["input_ids"], clear_inputs["input_ids"]):
                raise RuntimeError("shifted and acquired-view prompt token IDs differ")
            if not self.torch.equal(real_inputs["image_grid_thw"], clear_inputs["image_grid_thw"]):
                raise RuntimeError("shifted and acquired-view visual grids differ")
            clear_logprobs = self._score_tokens(clear_inputs, answer_ids)

        real_mean = _finite_mean(real_logprobs)
        blank_mean = _finite_mean(blank_logprobs)
        grounding = real_mean - blank_mean if real_logprobs else EMPTY_SCORE
        clear_mean = _finite_mean(clear_logprobs or []) if clear_logprobs is not None else None
        clear_grounding = (
            clear_mean - blank_mean if clear_mean is not None and real_logprobs else None
        )
        grid = tuple(int(value) for value in real_inputs["image_grid_thw"].flatten().tolist())
        return {
            "answer": answer,
            "answer_ids": tuple(int(value) for value in answer_ids.tolist()),
            "real_logprobs": tuple(real_logprobs),
            "blank_logprobs": tuple(blank_logprobs),
            "clear_logprobs": tuple(clear_logprobs) if clear_logprobs is not None else None,
            "mean_logprob": real_mean,
            "blank_logprob": blank_mean,
            "grounding_score": grounding,
            "clear_logprob": clear_mean,
            "clear_grounding_score": clear_grounding,
            "acquisition_score": (
                max(grounding, clear_grounding)
                if clear_grounding is not None
                else grounding
            ),
            "image_grid_thw": grid,
        }


class GenericVLAuditor(QwenVLAuditor):
    """AutoTransformers adapter for non-Qwen multimodal checkpoints.

    The selective-reliability protocol needs only deterministic generation and
    teacher-forced scores for the generated tokens. This adapter deliberately
    uses the same prompt, blank-image counterfactual, and score calculation as
    :class:`QwenVLAuditor`, while allowing model-specific Transformers classes.
    """

    def __init__(self, model_path: str | Path, *, max_new_tokens: int = 32) -> None:
        import torch
        from transformers import AutoProcessor, GenerationConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        model_name = Path(model_path).name.lower()
        processor_max_pixels = LLAVA_MAX_PIXELS if "llava" in model_name else MAX_PIXELS
        processor_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        try:
            self.processor = AutoProcessor.from_pretrained(
                str(model_path),
                min_pixels=MIN_PIXELS,
                max_pixels=processor_max_pixels,
                **processor_kwargs,
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(str(model_path), **processor_kwargs)

        model_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "device_map": "cuda",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "local_files_only": True,
            "_attn_implementation": (
                "sdpa" if "llava" in model_name or "smol" in model_name else "eager"
            ),
        }
        try:
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(
                str(model_path), **model_kwargs
            )
        except (AttributeError, ImportError, TypeError, ValueError):
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
        self.model = self.model.eval()
        self.device = next(self.model.parameters()).device
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_k=None,
            top_p=None,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_id,
        )
        self.model.generation_config = self.generation_config
        torch.cuda.reset_peak_memory_stats()

    def _processor_inputs(self, image: Image.Image, question: str) -> dict[str, Any]:
        encoded = self.processor.apply_chat_template(
            _conversation(image, question),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

    def _visual_signature(self, inputs: dict[str, Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
        signature: list[tuple[str, tuple[int, ...]]] = []
        for key in (
            "image_grid_thw",
            "pixel_values",
            "pixel_values_videos",
            "image_sizes",
        ):
            value = inputs.get(key)
            if value is not None and hasattr(value, "shape"):
                signature.append((key, tuple(int(size) for size in value.shape)))
        return tuple(signature)

    def _score_tokens(self, prompt_inputs: dict[str, Any], answer_ids: Any) -> list[float]:
        if answer_ids.numel() == 0:
            return []
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        model_inputs = dict(prompt_inputs)
        model_inputs["input_ids"] = self.torch.cat(
            [prompt_inputs["input_ids"], answer_ids.unsqueeze(0)], dim=1
        )
        model_inputs["attention_mask"] = self.torch.cat(
            [
                prompt_inputs["attention_mask"],
                self.torch.ones(
                    (1, answer_ids.shape[0]),
                    dtype=prompt_inputs["attention_mask"].dtype,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        with self.torch.inference_mode():
            output = self.model(**model_inputs, use_cache=False, return_dict=True)
        logits = output.logits[:, prompt_length - 1 : prompt_length + answer_ids.shape[0] - 1]
        values = (
            logits.double()
            .log_softmax(dim=-1)
            .gather(-1, answer_ids.view(1, -1, 1))
            .squeeze(0)
            .squeeze(-1)
        )
        return [float(value) for value in values.cpu().tolist()]

    def audit(
        self,
        image: Image.Image,
        question: str,
        *,
        clear_image: Image.Image | None = None,
        verify_generation_score: bool = False,
    ) -> dict[str, object]:
        real_inputs = self._processor_inputs(image, question)
        prompt_length = int(real_inputs["input_ids"].shape[1])
        with self.torch.inference_mode():
            output = self.model.generate(
                **real_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = output.sequences[0, prompt_length : prompt_length + len(output.scores)]
        selected_logprobs = [
            float(step_scores[0].double().log_softmax(dim=-1)[generated_ids[index]])
            for index, step_scores in enumerate(output.scores)
        ]
        special_ids = set(self.processor.tokenizer.all_special_ids)
        usable_positions = [
            index
            for index, token_id in enumerate(generated_ids.tolist())
            if token_id not in special_ids
        ]
        answer_ids = generated_ids[usable_positions]
        real_logprobs = [selected_logprobs[index] for index in usable_positions]
        answer = self.processor.decode(answer_ids, skip_special_tokens=True).strip()

        blank = Image.new("RGB", image.size, color=(127, 127, 127))
        try:
            blank_inputs = self._processor_inputs(blank, question)
        finally:
            blank.close()
        if not self.torch.equal(real_inputs["input_ids"], blank_inputs["input_ids"]):
            raise RuntimeError("real and blank prompt token IDs differ")
        if self._visual_signature(real_inputs) != self._visual_signature(blank_inputs):
            raise RuntimeError("real and blank visual signatures differ")
        blank_logprobs = self._score_tokens(blank_inputs, answer_ids)
        if len(blank_logprobs) != len(real_logprobs):
            raise RuntimeError("blank and real answer-token counts differ")

        if verify_generation_score and answer_ids.numel():
            forced_real = self._score_tokens(real_inputs, answer_ids)
            max_difference = max(
                abs(generated - forced)
                for generated, forced in zip(real_logprobs, forced_real, strict=True)
            )
            # Multimodal adapters may use a cached image path during
            # generation, producing a small numerical difference from a full
            # teacher-forced forward pass. Keep the sanity check while
            # allowing a documented margin for those implementations.
            if max_difference > GENERIC_GENERATION_SCORE_TOLERANCE:
                raise RuntimeError(
                    f"generation/teacher-forcing log-prob mismatch: {max_difference:.4f}"
                )

        clear_logprobs: list[float] | None = None
        if clear_image is not None:
            clear_inputs = self._processor_inputs(clear_image, question)
            if not self.torch.equal(real_inputs["input_ids"], clear_inputs["input_ids"]):
                raise RuntimeError("shifted and acquired-view prompt token IDs differ")
            if self._visual_signature(real_inputs) != self._visual_signature(clear_inputs):
                raise RuntimeError("shifted and acquired-view visual signatures differ")
            clear_logprobs = self._score_tokens(clear_inputs, answer_ids)

        real_mean = _finite_mean(real_logprobs)
        blank_mean = _finite_mean(blank_logprobs)
        grounding = real_mean - blank_mean if real_logprobs else EMPTY_SCORE
        clear_mean = _finite_mean(clear_logprobs or []) if clear_logprobs is not None else None
        clear_grounding = (
            clear_mean - blank_mean if clear_mean is not None and real_logprobs else None
        )
        grid = tuple(
            int(value)
            for value in real_inputs.get("image_grid_thw", self.torch.empty(0))
            .flatten()
            .tolist()
        )
        return {
            "answer": answer,
            "answer_ids": tuple(int(value) for value in answer_ids.tolist()),
            "real_logprobs": tuple(real_logprobs),
            "blank_logprobs": tuple(blank_logprobs),
            "clear_logprobs": tuple(clear_logprobs) if clear_logprobs is not None else None,
            "mean_logprob": real_mean,
            "blank_logprob": blank_mean,
            "grounding_score": grounding,
            "clear_logprob": clear_mean,
            "clear_grounding_score": clear_grounding,
            "acquisition_score": (
                max(grounding, clear_grounding)
                if clear_grounding is not None
                else grounding
            ),
            "image_grid_thw": grid,
        }


def build_auditor(model_key: str, model_path: Path, *, max_new_tokens: int = 32) -> Any:
    """Construct the pinned checkpoint adapter for ``model_key``."""

    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown model key: {model_key}")
    if model_key in QWEN_MODEL_KEYS:
        return QwenVLAuditor(model_path, max_new_tokens=max_new_tokens)
    return GenericVLAuditor(model_path, max_new_tokens=max_new_tokens)


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                completed.add(str(json.loads(line)["case_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def run_inference(
    rows: Iterable[ManifestRow],
    *,
    model_key: str,
    model_path: Path,
    output_path: Path,
    limit: int | None = None,
    commit_callback: Any | None = None,
) -> dict[str, float | int]:
    """Run checkpointed batch-one inference and return timing/memory statistics."""

    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown model key: {model_key}")
    model_id, revision = MODEL_SPECS[model_key]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_existing_ids(output_path)
    pending = [row for row in rows if row.case_id not in completed]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return {
            "written": 0,
            "natural_written": 0,
            "shift_written": 0,
            "load_seconds": 0.0,
            "inference_seconds": 0.0,
            "natural_seconds": 0.0,
            "shift_seconds": 0.0,
            "peak_memory_gb": 0.0,
        }

    tracker = create_tracker(
        model_key=model_key,
        model_id=model_id,
        model_revision=revision,
        config={
            "max_new_tokens": 32,
            "decoding": "greedy",
            "natural_rows": sum(row.regime == "natural" for row in pending),
            "shift_rows": sum(row.regime == "shift" for row in pending),
        },
    )
    load_started = time.perf_counter()
    auditor = build_auditor(model_key, model_path)
    load_seconds = time.perf_counter() - load_started
    tracker.log({"status": "loaded", "load_seconds": load_seconds})
    written = 0
    regime_counts = {"natural": 0, "shift": 0}
    regime_seconds = {"natural": 0.0, "shift": 0.0}
    inference_started = time.perf_counter()
    try:
        with output_path.open("a", encoding="utf-8", buffering=1) as handle:
            for row in pending:
                item_started = time.perf_counter()
                with Image.open(row.image_path) as source:
                    image = source.convert("RGB")
                clear_image: Image.Image | None = None
                try:
                    if row.clear_image_path:
                        with Image.open(row.clear_image_path) as clear_source:
                            clear_image = clear_source.convert("RGB")
                    audit = auditor.audit(
                        image,
                        row.question,
                        clear_image=clear_image,
                        verify_generation_score=written == 0,
                    )
                finally:
                    image.close()
                    if clear_image is not None:
                        clear_image.close()
                answer = str(audit["answer"])
                score = vqa_accuracy(answer, row.answers)
                prediction = Prediction(
                    case_id=row.case_id,
                    row_index=row.row_index,
                    model_key=model_key,
                    model_id=model_id,
                    model_revision=revision,
                    evaluation_split=row.evaluation_split,
                    regime=row.regime,
                    corruption=row.corruption,
                    source_case_id=row.source_case_id,
                    image_path=row.image_path,
                    clear_image_path=row.clear_image_path,
                    question=row.question,
                    reference_answers=row.answers,
                    answerable=row.answerable,
                    answer_type=row.answer_type,
                    generated_answer=answer,
                    normalized_answer=canonical_prediction(answer),
                    generated_token_ids=audit["answer_ids"],
                    real_token_logprobs=audit["real_logprobs"],
                    blank_token_logprobs=audit["blank_logprobs"],
                    clear_token_logprobs=audit["clear_logprobs"],
                    vqa_score=score,
                    entirely_wrong=score == 0.0,
                    false_answer_on_unanswerable=is_false_answer_on_unanswerable(
                        answer, row.answerable
                    ),
                    mean_logprob=float(audit["mean_logprob"]),
                    blank_logprob=float(audit["blank_logprob"]),
                    grounding_score=float(audit["grounding_score"]),
                    clear_logprob=(
                        float(audit["clear_logprob"])
                        if audit["clear_logprob"] is not None
                        else None
                    ),
                    clear_grounding_score=(
                        float(audit["clear_grounding_score"])
                        if audit["clear_grounding_score"] is not None
                        else None
                    ),
                    acquisition_score=float(audit["acquisition_score"]),
                    generated_tokens=len(audit["answer_ids"]),
                    image_grid_thw=audit["image_grid_thw"],
                )
                handle.write(json.dumps(asdict(prediction), sort_keys=True, allow_nan=False) + "\n")
                written += 1
                regime_counts[row.regime] += 1
                regime_seconds[row.regime] += time.perf_counter() - item_started
                if written % 100 == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    if commit_callback:
                        commit_callback()
                    tracker.log(
                        {
                            "rows_written": written,
                            "natural_written": regime_counts["natural"],
                            "shift_written": regime_counts["shift"],
                            "elapsed_seconds": time.perf_counter() - inference_started,
                            "peak_memory_gb": auditor.peak_memory_gb,
                        }
                    )
                    print(
                        f"{model_key}: wrote {written}/{len(pending)}; "
                        f"peak={auditor.peak_memory_gb:.1f} GiB",
                        flush=True,
                    )
        if commit_callback:
            commit_callback()
    finally:
        peak_memory = auditor.peak_memory_gb
        auditor.close()
        tracker.finish(
            summary={
                "rows_written": written,
                "natural_written": regime_counts["natural"],
                "shift_written": regime_counts["shift"],
                "load_seconds": load_seconds,
                "inference_seconds": time.perf_counter() - inference_started,
                "peak_memory_gb": peak_memory,
            }
        )
    return {
        "written": written,
        "natural_written": regime_counts["natural"],
        "shift_written": regime_counts["shift"],
        "load_seconds": load_seconds,
        "inference_seconds": time.perf_counter() - inference_started,
        "natural_seconds": regime_seconds["natural"],
        "shift_seconds": regime_seconds["shift"],
        "peak_memory_gb": peak_memory,
    }
