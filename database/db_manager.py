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

    # Ensure tax_type column exists for backwards compatibility
    try:
        cursor.execute("ALTER TABLE receipts ADD COLUMN tax_type TEXT;")
    except sqlite3.OperationalError:
        # Column already exists, safe to ignore
        pass
    
    # Ensure quantity column exists in line_items for backwards compatibility
    try:
        cursor.execute("ALTER TABLE line_items ADD COLUMN quantity INTEGER DEFAULT 1;")
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
        quantity INTEGER DEFAULT 1,
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
        "tax_type": "...",
        "items": [
            {
                "japanese_name": "...",
                "english_name": "...",
                "price": 0,
                "quantity": 1,
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
        tax_type = receipt_data.get("tax_type", "included")
        # Use parsed date from receipt if available, else current date/time
        date_str = receipt_data.get("date") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        savings_advice = receipt_data.get("savings_advice", "")
        raw_json_str = json.dumps(receipt_data, ensure_ascii=False)
        
        # 1. Insert Receipt
        cursor.execute("""
        INSERT INTO receipts (store_name, total_amount, tax_amount, tax_type, date, savings_advice, raw_ocr_or_json, image_file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (store_name, total_amount, tax_amount, tax_type, date_str, savings_advice, raw_json_str, image_path))
        
        receipt_id = cursor.lastrowid
        logger.info(f"Receipt record inserted with ID: {receipt_id}")
        
        # 2. Insert Line Items
        items = receipt_data.get("items", [])
        for item in items:
            japanese_name = item.get("japanese_name", "")
            english_name = item.get("english_name", "")
            price = item.get("price", 0)
            category = item.get("category", "Other")
            confidence = item.get("confidence", 1.0)
            try:
                quantity = int(item.get("quantity") or 1)
            except Exception:
                quantity = 1
            if quantity < 1:
                quantity = 1
            
            cursor.execute("""
            INSERT INTO line_items (receipt_id, item_name, english_name, price, category, confidence_score, quantity)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (receipt_id, japanese_name, english_name, price, category, confidence, quantity))
            
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


def delete_receipt_records(date_str: str, store_name: str) -> bool:
    """Delete a receipt matching the date and store name from SQLite database."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM receipts WHERE store_name = ? AND date = ?", (store_name, date_str))
        row = cursor.fetchone()
        if row:
            receipt_id = row[0]
            cursor.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))
            conn.commit()
            logger.info(f"Deleted receipt {receipt_id} from SQLite.")
            conn.close()
            return True
        conn.close()
        return False
    except Exception as e:
        logger.error(f"Error deleting receipt from SQLite: {e}")
        return False


def delete_single_item_records(date_str: str, store_name: str, jp_name: str, eng_name: str) -> bool:
    """Delete a single line item matching date, store, and item names from SQLite database."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find receipt_id
        cursor.execute("SELECT id FROM receipts WHERE store_name = ? AND date = ?", (store_name, date_str))
        row = cursor.fetchone()
        if row:
            receipt_id = row[0]
            # Delete from line_items where receipt_id matches, and either japanese_name or english_name matches
            cursor.execute("""
                DELETE FROM line_items 
                WHERE receipt_id = ? 
                AND (item_name = ? OR english_name = ?)
            """, (receipt_id, jp_name, eng_name))
            
            # Get tax_amount and tax_type
            cursor.execute("SELECT tax_amount, tax_type FROM receipts WHERE id = ?", (receipt_id,))
            tax_row = cursor.fetchone()
            tax_amount = tax_row[0] or 0 if tax_row else 0
            tax_type = tax_row[1] or "included" if tax_row else "included"
            
            # Get remaining items to recalculate total
            cursor.execute("SELECT price, quantity, category FROM line_items WHERE receipt_id = ?", (receipt_id,))
            remaining_items = cursor.fetchall()
            
            subtotal = 0
            discount = 0
            for price, quantity, category in remaining_items:
                cat_lower = (category or "").lower()
                price_val = float(price or 0)
                qty_val = int(quantity or 1)
                
                if cat_lower == "change":
                    continue
                elif cat_lower == "discount":
                    discount += price_val * qty_val
                else:
                    subtotal += price_val * qty_val
            
            new_total = subtotal - discount
            if tax_type == "excluded":
                new_total += tax_amount
                
            cursor.execute("UPDATE receipts SET total_amount = ? WHERE id = ?", (new_total, receipt_id))
            
            conn.commit()
            logger.info(f"Deleted item '{jp_name or eng_name}' from receipt {receipt_id} in SQLite. New total: {new_total}")
            conn.close()
            return True
        conn.close()
        return False
    except Exception as e:
        logger.error(f"Error deleting single item from SQLite: {e}")
        return False
