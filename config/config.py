"""Central configuration for the Japanese receipt OCR pipeline.

Every path and tunable lives here so the rest of the codebase never
hardcodes anything. Override any value with an environment variable of the
same name (e.g. RECEIPT_SEED=7) or by editing config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

# Project root = the directory that contains this config package's parent.
ROOT = Path(__file__).resolve().parents[1]

DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
DATABASE_DIR = ROOT / "database"
LOGS_DIR = ROOT / "outputs" / "logs"

for _d in (DATASETS_DIR, MODELS_DIR, OUTPUTS_DIR, DATABASE_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DetectionConfig:
    """Classical crop/deskew parameters (config-driven, no model)."""

    gaussian_kernel: int = 5
    canny_low: int = 75
    canny_high: int = 200
    min_area_ratio: float = 0.15  # contour must cover >=15% of the image
    approx_epsilon_ratio: float = 0.02
    # If the classical detector fails and a fallback detector is trained,
    # this path holds it. None => no learned fallback available.
    fallback_model_path: str | None = None
    output_max_side: int = 1600  # longest side of the warped receipt


@dataclass
class OCRConfig:
    """OCR engine + fine-tuning settings."""

    engine: str = "easyocr"  # "easyocr" or "paddleocr"
    languages: tuple[str, ...] = ("ja", "en")
    gpu: bool = False
    # Fine-tuned recognition weights. If present they are loaded; otherwise
    # the pretrained bundled recognizer is used.
    recognizer_weights: str | None = None
    recognizer_network: str = "standard"  # easyocr custom-model name
    min_confidence: float = 0.30  # drop OCR lines below this


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    num_workers: int = 0  # 0 is safest on Windows
    val_split: float = 0.1
    early_stop_patience: int = 5
    checkpoint_every: int = 1


@dataclass
class ParseConfig:
    """Rule-based parsing + embedding categorization."""

    # Regex fragments are assembled in inference/parse.py.
    currency_symbols: tuple[str, ...] = ("¥", "円")
    embedding_model: str = "intfloat/multilingual-e5-small"
    category_threshold: float = 0.55  # below => Other/Uncategorized
    categories_file: str = str(ROOT / "config" / "categories.yaml")
    use_bert_correction: bool = False  # fill-mask cleanup is opt-in (heavy)
    bert_model: str = "cl-tohoku/bert-base-japanese-v3"


@dataclass
class Config:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parse: ParseConfig = field(default_factory=ParseConfig)

    datasets_dir: str = str(DATASETS_DIR)
    models_dir: str = str(MODELS_DIR)
    outputs_dir: str = str(OUTPUTS_DIR)
    database_path: str = str(DATABASE_DIR / "receipts.db")
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        """Load config.yaml if present, then apply env overrides."""
        cfg = cls()
        path = Path(path) if path else ROOT / "config" / "config.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            cfg = cls._merge(cfg, data)
        cfg._apply_env()
        return cfg

    @staticmethod
    def _merge(cfg: "Config", data: dict) -> "Config":
        for section in ("detection", "ocr", "training", "parse"):
            if section in data and isinstance(data[section], dict):
                obj = getattr(cfg, section)
                for k, v in data[section].items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
        for k in ("datasets_dir", "models_dir", "outputs_dir",
                  "database_path", "log_level"):
            if k in data:
                setattr(cfg, k, data[k])
        return cfg

    def _apply_env(self) -> None:
        if "RECEIPT_SEED" in os.environ:
            self.training.seed = int(os.environ["RECEIPT_SEED"])
        if "RECEIPT_LOG_LEVEL" in os.environ:
            self.log_level = os.environ["RECEIPT_LOG_LEVEL"]
        if "RECEIPT_GPU" in os.environ:
            self.ocr.gpu = os.environ["RECEIPT_GPU"] == "1"

    def to_dict(self) -> dict:
        return asdict(self)


# A module-level singleton most code can just import.
settings = Config.load()
