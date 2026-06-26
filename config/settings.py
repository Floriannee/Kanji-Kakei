import os
import logging

# Setup default configurations
DB_FILE = os.path.join("database", "kakei.db")
OUTPUT_DIR = "outputs"

# Configure logging format globally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KanjiKakei")
