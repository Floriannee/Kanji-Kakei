# Training notes

This project doesn't train anything from scratch. The only optional
training step is fine-tuning PaddleOCR's recognition module on the CORD
dataset, to adapt it from natural-scene text to receipt-specific fonts
and layouts.

This isn't implemented yet. When it is, it should run as a notebook on a
free cloud GPU (Google Colab or Kaggle Notebooks both give free NVIDIA T4
access), not locally -- a laptop without a dedicated GPU can run the
*finished* model for inference just fine, but fine-tuning it would be
impractically slow.

Workflow once this is built:
1. Run the fine-tuning notebook on Colab/Kaggle.
2. Download the resulting model weights.
3. Drop them into this project (e.g. `models/ocr/`) and point `src/ocr.py`
   at them.
4. Everything downstream runs locally on CPU as normal.
