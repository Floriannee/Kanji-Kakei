# 漢字家計 Kanji-Kakei 🧾💡
> **Japanese Receipt Reader & AI Financial Advisor for Foreign Students**
>
> A modern, Fluent Windows 11 desktop application built using `customtkinter`, `opencv`, and LLM APIs to help international students in Japan scan Japanese retail receipts, translate itemized expenditures, manage monthly budgets, and get context-aware financial advice.

---

## 🌟 Key Features

1. **OCR Japanese Receipt Scanning**
   - Upload receipt images (PNG, JPG, JPEG, BMP).
   - Automatically extract store name, total amount, tax amount, savings advice, and itemized lists with translations and cultural explanations.

2. **Classical CV Deskewing & Warp**
   - Preprocesses images using OpenCV (Grayscale conversion, Gaussian blurring, Canny edge detection, and a 4-point perspective warp `cv2.warpPerspective`).
   - If edge detection fails (e.g. low contrast), it safely falls back to the original image without crashing.

3. **2-Tier Multimodal LLM Pipeline**
   - Resolves abbreviations (e.g., `ｱｰﾓﾝﾄﾞﾁｮｺ` -> `明治 アーモンドチョコレート`).
   - Runs a resilient 2-tier fallback check to ensure the application never crashes under API rate limits:
     * **TIER 1 (Primary)**: Groq API using model `meta-llama/llama-4-scout-17b-16e-instruct` (Fast and Free).
     * **TIER 2 (Fallback)**: Offline Simulation Mode (returns a mock FamilyMart response block).

4. **Robust Database Storage**
   - Persists receipts and item lists inside local SQLite database (`database/kakei.db`) with cascading deletion support.

5. **Stopwatch Metrics**
   - Built-in real-time stopwatch ticking every 50ms to demonstrate execution times under 3 seconds live.

6. **Pandas CSV Exporting**
   - Single-click export of parsed items to the `outputs/` folder with proper Japanese encoding formatting for Excel.

---

## 🛠️ Codebase Architecture

The project conforms to a clean, flat modular structure:
- [main.py](file:///C:/Machine%20Intelligence/Receipt/main.py): Renders the customtkinter GUI, stopwatch threads, image rendering, and exports.
- [config/settings.py](file:///C:/Machine%20Intelligence/Receipt/config/settings.py): Global config variables (database path, outputs folder, API models).
- [database/db_manager.py](file:///C:/Machine%20Intelligence/Receipt/database/db_manager.py): SQLite connections, schema initialization, and receipt insertions.
- [inference/pipeline.py](file:///C:/Machine%20Intelligence/Receipt/inference/pipeline.py): Houses `ReceiptParser` class executing the LLM API multimodal JSON requests with fallbacks.
- [utils/image_processing.py](file:///C:/Machine%20Intelligence/Receipt/utils/image_processing.py): OpenCV warp algorithms, point ordering math, and PIL converters.
- [outputs/](file:///C:/Machine%20Intelligence/Receipt/outputs/): Stores exported CSV receipts.
- [tests/test_components.py](file:///C:/Machine%20Intelligence/Receipt/tests/test_components.py): Unit tests for point math, fallbacks, database transactions, and model mocks.

---

## 🚀 Installation & Running

### Prerequisites
1.  **Configure API Key**: Set your Groq API key as an environment variable in your terminal:
    *   **PowerShell**:
        ```powershell
        $env:GROQ_API_KEY="your_groq_key"
        ```
    *   **Git Bash**:
        ```bash
        export GROQ_API_KEY="your_groq_key"
        ```
    *(If no API key is set or the API fails, the application runs in Simulation Mode, returning a realistic pre-coded FamilyMart response so you can still test all functions offline).*

### Execution Instructions
1.  **Activate virtual environment and install dependencies**:
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\pip install -r requirements.txt
    ```
2.  **Start the GUI**:
    *   **Git Bash**: Run `./run.sh`
    *   **CMD / PowerShell**: Double-click `run.bat` or run `.\run.bat`

---

## 📸 Testing with Your Own Pictures

1.  Place your receipt image (PNG, JPG, or JPEG) anywhere on your PC.
2.  Launch Kanji-Kakei using `./run.sh` or `run.bat`.
3.  Click the **📁 Upload Receipt** button on the left sidebar and select your file.
4.  The preprocessed (deskewed) image will display in the preview window on the left.
5.  Click **⚙️ Process Receipt** to start parsing. The stopwatch will tick, and the final itemized JSON ledger list and savings advice will render in the text box.
6.  Check the terminal logs to see which provider succeeded:
    `[Pipeline]  Success via Groq`
7.  Click **📥 Export to CSV** to save the table to the `outputs/` folder.
