import sqlite3
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV_FILE = "database/kakei_items.csv"
DB_FILE = "database/kakei.db"
RECEIPTS_DIR = "receipts"

def clear():
    # 1. Clear SQLite tables
    if os.path.exists(DB_FILE):
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("DELETE FROM line_items")
            c.execute("DELETE FROM receipts")
            try:
                c.execute("DELETE FROM sqlite_sequence")
            except sqlite3.OperationalError:
                pass
            conn.commit()
            conn.close()
            print("SQLite database tables cleared.")
        except Exception as e:
            print(f"Error clearing SQLite: {e}")
            
    # 2. Overwrite CSV ledger
    if os.path.exists(CSV_FILE):
        try:
            from database.csv_manager import FIELDNAMES
            with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
            print("CSV ledger cleared.")
        except Exception as e:
            print(f"Error clearing CSV: {e}")
            
    # 3. Clear receipts images folder
    if os.path.exists(RECEIPTS_DIR):
        try:
            for f in os.listdir(RECEIPTS_DIR):
                f_path = os.path.join(RECEIPTS_DIR, f)
                if os.path.isfile(f_path):
                    os.remove(f_path)
            print("Receipts images folder cleared.")
        except Exception as e:
            print(f"Error clearing receipts folder: {e}")
            
    # 4. Regenerate dashboard HTML
    try:
        from main import generate_html_dashboard
        generate_html_dashboard()
        print("HTML dashboard regenerated to empty state.")
    except Exception as e:
        print(f"Error regenerating HTML: {e}")

if __name__ == "__main__":
    clear()
