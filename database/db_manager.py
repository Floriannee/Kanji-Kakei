import os
import sqlite3
import json
import logging
from datetime import datetime
from config.settings import DB_FILE

# Initialize logger
logger = logging.getLogger("KanjiKakei.Database")

def get_connection():
    """Establish and return a connection to the SQLite database."""
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    # Enable foreign keys support
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initialize the SQLite database and create required tables."""
    logger.info(f"Initializing database at path: {DB_FILE}")
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Create Receipts Table (with tax_amount)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS receipts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        store_name TEXT,
        total_amount INTEGER,
        tax_amount INTEGER,
        date TEXT,
        savings_advice TEXT,
        raw_ocr_or_json TEXT,
        image_file_path TEXT
    );
    """)
    
    # Ensure tax_amount column exists for backwards compatibility
    try:
        cursor.execute("ALTER TABLE receipts ADD COLUMN tax_amount INTEGER;")
    except sqlite3.OperationalError:
        # Column already exists, safe to ignore
        pass
    
    # 2. Create Line Items Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS line_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        receipt_id INTEGER,
        item_name TEXT,
        english_name TEXT,
        price INTEGER,
        category TEXT,
        confidence_score REAL,
        FOREIGN KEY (receipt_id) REFERENCES receipts(id) ON DELETE CASCADE
    );
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database tables initialized successfully.")

def insert_receipt(receipt_data: dict, image_path: str = "") -> int:
    """
    Insert a fully parsed receipt record along with its line items.
    
    receipt_data must follow the LLM JSON schema:
    {
        "store_name": "...",
        "total_amount": 0,
        "tax_amount": 0,
        "items": [
            {
                "japanese_name": "...",
                "english_name": "...",
                "price": 0,
                "category": "...",
                "confidence": 0.95
            }
        ],
        "savings_advice": "..."
    }
    """
    logger.info(f"Inserting new receipt for store: {receipt_data.get('store_name', 'Unknown')}")
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get values
        store_name = receipt_data.get("store_name", "Unknown Store")
        total_amount = receipt_data.get("total_amount", 0)
        tax_amount = receipt_data.get("tax_amount", 0)
        # Use current date if no date is parsed
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        savings_advice = receipt_data.get("savings_advice", "")
        raw_json_str = json.dumps(receipt_data, ensure_ascii=False)
        
        # 1. Insert Receipt
        cursor.execute("""
        INSERT INTO receipts (store_name, total_amount, tax_amount, date, savings_advice, raw_ocr_or_json, image_file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (store_name, total_amount, tax_amount, date_str, savings_advice, raw_json_str, image_path))
        
        receipt_id = cursor.lastrowid
        logger.info(f"Receipt record inserted with ID: {receipt_id}")
        
        # 2. Insert Line Items
        items = receipt_data.get("items", [])
        for item in items:
            japanese_name = item.get("japanese_name", "")
            english_name = item.get("english_name", "")
            price = item.get("price", 0)
            category = item.get("category", "Miscellaneous")
            confidence = item.get("confidence", 1.0)
            
            cursor.execute("""
            INSERT INTO line_items (receipt_id, item_name, english_name, price, category, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (receipt_id, japanese_name, english_name, price, category, confidence))
            
        conn.commit()
        logger.info(f"Successfully inserted {len(items)} line items for receipt ID: {receipt_id}")
        return receipt_id
        
    except Exception as db_err:
        conn.rollback()
        logger.error(f"Failed to insert receipt record: {db_err}")
        raise db_err
    finally:
        conn.close()

def get_recent_receipts(limit: int = 10) -> list:
    """Retrieve recent receipts from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT id, store_name, total_amount, date, savings_advice, image_file_path
        FROM receipts
        ORDER BY id DESC
        LIMIT ?;
        """, (limit,))
        rows = cursor.fetchall()
        
        receipts = []
        for row in rows:
            receipts.append({
                "id": row[0],
                "store_name": row[1],
                "total_amount": row[2],
                "date": row[3],
                "savings_advice": row[4],
                "image_file_path": row[5]
            })
        return receipts
    finally:
        conn.close()
