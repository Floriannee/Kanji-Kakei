"""Step 3: OCR. Wraps EasyOCR (default) or PaddleOCR behind one interface.

Both engines bundle a pretrained detector + recognizer. If
``recognizer_weights`` is set in OCRConfig, the EasyOCR path loads those
fine-tuned weights instead of the stock recognizer (see
training/train_ocr.py).

The wrapper returns a uniform list of OCRLine objects sorted top-to-bottom
so the parser downstream never has to know which engine produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from config import settings
from config.config import OCRConfig
from utils.exceptions import OCRFailureError
from utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class OCRLine:
    text: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h (axis-aligned)

    @property
    def cy(self) -> int:
        return self.box[1] + self.box[3] // 2


class OCREngine:
    """Lazy-loading OCR engine. Construct once and reuse across images."""

    def __init__(self, cfg: OCRConfig | None = None):
        self.cfg = cfg or settings.ocr
        self._reader = None  # built on first use

    # --- engine construction -------------------------------------------------
    def _build(self):
        if self.cfg.engine == "easyocr":
            return self._build_easyocr()
        if self.cfg.engine == "paddleocr":
            return self._build_paddleocr()
        raise ValueError(f"Unknown OCR engine: {self.cfg.engine}")

    def _build_easyocr(self):
        import easyocr

        kwargs = dict(
            lang_list=list(self.cfg.languages),
            gpu=self.cfg.gpu,
        )
        if self.cfg.recognizer_weights:
            # Custom fine-tuned recognizer: EasyOCR loads a network from
            # model_storage_directory by recog_network name.
            from pathlib import Path

            weights = Path(self.cfg.recognizer_weights)
            kwargs.update(
                model_storage_directory=str(weights.parent),
                user_network_directory=str(weights.parent),
                recog_network=self.cfg.recognizer_network,
            )
            logger.info("EasyOCR using fine-tuned recognizer: %s", weights)
        reader = easyocr.Reader(**kwargs)
        return reader

    def _build_paddleocr(self):
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x: lang controls bundled models; 'japan' covers JP+EN.
        reader = PaddleOCR(lang="japan", use_textline_orientation=True)
        return reader

    @property
    def reader(self):
        if self._reader is None:
            logger.info("Initializing OCR engine '%s'", self.cfg.engine)
            self._reader = self._build()
        return self._reader

    # --- inference -----------------------------------------------------------
    def read(self, image: np.ndarray) -> list[OCRLine]:
        """Run OCR on a (cropped, deskewed) BGR image."""
        if self.cfg.engine == "easyocr":
            lines = self._read_easyocr(image)
        else:
            lines = self._read_paddleocr(image)

        lines = [l for l in lines if l.confidence >= self.cfg.min_confidence]
        if not lines:
            raise OCRFailureError("OCR produced no lines above the confidence floor.")

        lines.sort(key=lambda l: l.cy)  # top-to-bottom reading order
        logger.info("OCR produced %d lines", len(lines))
        return lines

    def _read_easyocr(self, image: np.ndarray) -> list[OCRLine]:
        # EasyOCR accepts BGR numpy arrays directly.
        results = self.reader.readtext(image, detail=1, paragraph=False)
        out: list[OCRLine] = []
        for box, text, conf in results:
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            out.append(OCRLine(text=text, confidence=float(conf), box=(x, y, w, h)))
        return out

    def _read_paddleocr(self, image: np.ndarray) -> list[OCRLine]:
        result = self.reader.predict(image)
        out: list[OCRLine] = []
        # PaddleOCR 3.x returns a list of dict-like result objects.
        for page in result:
            texts = page.get("rec_texts") or page.get("rec_text") or []
            scores = page.get("rec_scores") or page.get("rec_score") or []
            polys = page.get("rec_polys") or page.get("dt_polys") or []
            for text, score, poly in zip(texts, scores, polys):
                pts = np.array(poly).reshape(-1, 2)
                x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
                w = int(pts[:, 0].max()) - x
                h = int(pts[:, 1].max()) - y
                out.append(
                    OCRLine(text=str(text), confidence=float(score), box=(x, y, w, h))
                )
        return out


@lru_cache(maxsize=1)
def get_default_engine() -> OCREngine:
    """Shared engine instance so scripts don't reload weights repeatedly."""
    return OCREngine()
