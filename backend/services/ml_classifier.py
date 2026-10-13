"""
PromptShield — ML Classifier Service
======================================

Thin async wrapper around ml.inference.InjectionPredictor.
Decouples the FastAPI service layer from ML internals so the
model can be swapped or version-gated without touching API code.

Lifecycle
---------
- ``load()`` is called once at FastAPI startup (lifespan handler).
- ``predict()`` / ``predict_batch()`` are called per-request.
- Heavy model forward pass runs on the calling thread; for very high
  concurrency, move to a threadpool via ``asyncio.to_thread()``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("promptshield.services.ml_classifier")

_DEFAULT_MODEL_PATH = Path("models/fine_tuned/promptshield_v2/best_model")
if not _DEFAULT_MODEL_PATH.exists() or not (_DEFAULT_MODEL_PATH / "model.safetensors").exists():
    _DEFAULT_MODEL_PATH = Path("models/fine_tuned/promptshield_v1/best_model")


class MLClassifierService:
    """
    Async wrapper around the fine-tuned transformer classifier.

    Attributes
    ----------
    model_path : Path to the saved model directory.
    threshold  : Decision threshold for injection label (default 0.5).
    _loaded    : Whether the model is ready for inference.
    """

    _instance: "MLClassifierService | None" = None

    def __init__(
        self,
        model_path: Path | str = _DEFAULT_MODEL_PATH,
        threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._predictor: Any = None  # ml.inference.InjectionPredictor
        self._loaded = False

    # ── Singleton ──────────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        model_path: Path | str = _DEFAULT_MODEL_PATH,
        threshold: float = 0.5,
    ) -> "MLClassifierService":
        if cls._instance is None:
            cls._instance = cls(model_path=model_path, threshold=threshold)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Used in tests to get a fresh instance."""
        cls._instance = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def load(self) -> None:
        """
        Load the model and tokenizer from disk.

        Runs the blocking I/O in a thread executor to avoid blocking
        the event loop during startup.
        """
        if self._loaded:
            return

        if not self.model_path.exists():
            log.warning(
                f"ML model checkpoint not found at {self.model_path}. "
                "The ML classifier layer will be DISABLED. "
                "Run 'python -m ml.train' to train and save a model."
            )
            return

        log.info(f"Loading ML classifier from: {self.model_path} ...")

        def _load_sync() -> Any:
            # Lazy import — keeps FastAPI startup fast when torch unavailable
            from ml.inference import InjectionPredictor  # noqa: PLC0415
            predictor = InjectionPredictor(
                model_path=self.model_path,
                threshold=self.threshold,
            )
            predictor.load()
            return predictor

        try:
            self._predictor = await asyncio.to_thread(_load_sync)
            self._loaded = True
            log.info("ML classifier loaded successfully.")
        except Exception as exc:
            log.error(f"ML classifier failed to load: {exc}. Layer will be disabled.")
            self._predictor = None
            self._loaded = False

    # ── Inference ──────────────────────────────────────────────────────────

    async def predict(self, prompt: str) -> dict[str, Any]:
        """
        Classify a single prompt.

        Returns
        -------
        dict with keys:
            layer          : "ml_classifier"
            triggered      : bool
            confidence     : float (0–1)
            injection_prob : float
            benign_prob    : float
            details        : str
            available      : bool  (False if model not loaded)
        """
        if not self._loaded or self._predictor is None:
            return {
                "layer": "ml_classifier",
                "triggered": False,
                "confidence": 0.0,
                "injection_prob": 0.0,
                "benign_prob": 1.0,
                "details": "model_not_loaded",
                "available": False,
            }

        t0 = time.perf_counter()
        result = await asyncio.to_thread(self._predictor.predict_single, prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "layer": "ml_classifier",
            "triggered": result["is_injection"],
            "confidence": result["confidence"],
            "injection_prob": result["injection_probability"],
            "benign_prob": result["benign_probability"],
            "details": (
                f"injection_prob={result['injection_probability']:.3f} "
                f"threshold={result['threshold_used']} "
                f"latency={elapsed_ms:.1f}ms"
            ),
            "available": True,
        }

    async def predict_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        """Batch classify multiple prompts."""
        if not self._loaded or self._predictor is None:
            return [
                {
                    "layer": "ml_classifier",
                    "triggered": False,
                    "confidence": 0.0,
                    "injection_prob": 0.0,
                    "benign_prob": 1.0,
                    "details": "model_not_loaded",
                    "available": False,
                }
                for _ in prompts
            ]

        raw_results = await asyncio.to_thread(self._predictor.predict_batch, prompts)
        return [
            {
                "layer": "ml_classifier",
                "triggered": r["is_injection"],
                "confidence": r["confidence"],
                "injection_prob": r["injection_probability"],
                "benign_prob": r["benign_probability"],
                "details": f"injection_prob={r['injection_probability']:.3f}",
                "available": True,
            }
            for r in raw_results
        ]

    @property
    def is_available(self) -> bool:
        return self._loaded and self._predictor is not None
