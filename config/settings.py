import os
import logging

# Setup default configurations
DB_FILE = os.path.join("database", "kakei.db")
# Continually-updated CSV ledger that backs the main "stored information" list in the GUI
CSV_FILE = os.path.join("database", "kakei_items.csv")
OUTPUT_DIR = "outputs"

# Configure logging format globally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KanjiKakei")
