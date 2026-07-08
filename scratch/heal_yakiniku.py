import sqlite3
import csv
import json
import os

CSV_FILE = "database/kakei_items.csv"
DB_FILE = "database/kakei.db"

def heal():
    # 1. Update SQLite
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute("SELECT id, raw_ocr_or_json, image_file_path FROM receipts WHERE store_name LIKE '%Ikuta%'")
    receipts = c.fetchall()
    
    yakiniku_ids = []
    for r_id, raw_json, img_path in receipts:
        yakiniku_ids.append(r_id)
        
        try:
            raw_data = json.loads(raw_json)
        except Exception:
            raw_data = {}
            
        raw_data["total_amount"] = 38397
        raw_data["tax_amount"] = 2603
        
        items = raw_data.get("items", [])
        items = [it for it in items if it.get("category") != "Change"]
        
        items.append({
            "japanese_name": "お預かり",
            "english_name": "Received",
            "category": "Change",
            "price": 39000,
            "quantity": 1,
            "note": "Amount received from customer"
        })
        items.append({
            "japanese_name": "お釣り",
            "english_name": "Change",
            "category": "Change",
            "price": 603,
            "quantity": 1,
            "note": "Change given to customer"
        })
        raw_data["items"] = items
        
        new_raw_json = json.dumps(raw_data)
        c.execute("UPDATE receipts SET total_amount = 38397, tax_amount = 2603, raw_ocr_or_json = ? WHERE id = ?", (new_raw_json, r_id))
        
        c.execute("DELETE FROM line_items WHERE receipt_id = ? AND category = 'Change'", (r_id,))
        
        c.execute("""
            INSERT INTO line_items (receipt_id, item_name, english_name, price, category, confidence_score, quantity)
            VALUES (?, 'お預かり', 'Received', 39000, 'Change', 1.0, 1)
        """, (r_id,))
        
        c.execute("""
            INSERT INTO line_items (receipt_id, item_name, english_name, price, category, confidence_score, quantity)
            VALUES (?, 'お釣り', 'Change', 603, 'Change', 1.0, 1)
        """, (r_id,))
        
    conn.commit()
    conn.close()
    print(f"Updated SQLite receipts: {yakiniku_ids}")
    
    # 2. Update CSV file
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
            
        new_rows = []
        receipt_groups = {}
        for r in rows:
            img = r.get("image_path") or ""
            dt = r.get("date") or ""
            store = r.get("store_name") or ""
            group_key = (img, dt, store)
            if group_key not in receipt_groups:
                receipt_groups[group_key] = []
            receipt_groups[group_key].append(r)
            
        import uuid
        for group_key, group_rows in receipt_groups.items():
            img, dt, store = group_key
            if "Ikuta" in store:
                group_rows = [r for r in group_rows if r.get("category") != "Change"]
                
                for r in group_rows:
                    r["receipt_total"] = "38397"
                    r["tax_amount"] = "2603"
                    
                row_received = {
                    "id": uuid.uuid4().hex,
                    "date": dt,
                    "store_name": store,
                    "japanese_name": "お預かり",
                    "english_name": "Received",
                    "category": "Change",
                    "price": "39000",
                    "quantity": "1",
                    "note": "Amount received from customer",
                    "receipt_total": "38397",
                    "tax_amount": "2603",
                    "tax_type": "excluded",
                    "savings_advice": group_rows[0].get("savings_advice") if group_rows else "",
                    "image_path": img
                }
                row_change = {
                    "id": uuid.uuid4().hex,
                    "date": dt,
                    "store_name": store,
                    "japanese_name": "お釣り",
                    "english_name": "Change",
                    "category": "Change",
                    "price": "603",
                    "quantity": "1",
                    "note": "Change given to customer",
                    "receipt_total": "38397",
                    "tax_amount": "2603",
                    "tax_type": "excluded",
                    "savings_advice": group_rows[0].get("savings_advice") if group_rows else "",
                    "image_path": img
                }
                group_rows.append(row_received)
                group_rows.append(row_change)
                
            new_rows.extend(group_rows)
            
        with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
            
        print("Updated CSV ledger.")

if __name__ == "__main__":
    heal()
