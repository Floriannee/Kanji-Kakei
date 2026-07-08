import os
import csv
import uuid
import logging
from datetime import datetime

from config.settings import CSV_FILE

# Initialize logger
logger = logging.getLogger("KanjiKakei.CSV")

FIELDNAMES = [
    "id",
    "date",
    "store_name",
    "japanese_name",
    "english_name",
    "category",
    "price",
    "quantity",
    "note",
    "receipt_total",
    "tax_amount",
    "tax_type",
    "savings_advice",
    "image_path",
]


def upgrade_csv_if_needed():
    if not os.path.exists(CSV_FILE):
        return
    try:
        with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return
        header = rows[0]
        if "quantity" not in header:
            logger.info("Upgrading CSV ledger to include quantity column...")
            new_rows = [FIELDNAMES]
            for row in rows[1:]:
                if len(row) == 11:
                    new_row = row[:6] + ["1"] + [row[6]] + [row[7]] + [row[8]] + ["included"] + row[9:]
                    new_rows.append(new_row)
                elif len(row) == 12:
                    new_row = row[:6] + ["1"] + row[6:]
                    new_rows.append(new_row)
                elif len(row) == 13:
                    new_rows.append(row)
                else:
                    new_row = row + [""] * (13 - len(row))
                    new_rows.append(new_row)
            with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerows(new_rows)
            logger.info("CSV ledger upgraded successfully.")
    except Exception as e:
        logger.error(f"Error upgrading CSV: {e}")


def _write_rows(rows: list):
    """Overwrite the CSV ledger with the given rows (list of dicts), including the header."""
    with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def init_csv():
    """
    Create the CSV ledger with a header row if it does not already exist.
    Also migrates ledgers saved by an earlier version of the app that predates
    the "id" column, since unique row ids are required for selecting/deleting
    individual rows in the UI.
    """
    parent_dir = os.path.dirname(CSV_FILE)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(CSV_FILE):
        upgrade_csv_if_needed()
    else:
        _write_rows([])
        logger.info(f"Created new CSV ledger at path: {CSV_FILE}")
        return

    with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or []
        if "id" in existing_fieldnames:
            return  # Already up to date, nothing to migrate
        legacy_rows = list(reader)

    logger.info("Migrating legacy CSV ledger (saved before row ids existed) to add unique row ids.")
    for row in legacy_rows:
        row["id"] = uuid.uuid4().hex

    _write_rows(legacy_rows)
    logger.info(f"Successfully migrated {len(legacy_rows)} legacy rows with unique ids.")


def append_items_to_csv(receipt_data: dict, image_path: str = "") -> int:
    """
    Append parsed receipt line items to the CSV ledger.
    Returns the number of rows written.
    """
    init_csv()
    store_name = receipt_data.get("store_name", "Unknown Store")
    date_str = receipt_data.get("date") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_amount = receipt_data.get("total_amount", 0)
    tax_amount = receipt_data.get("tax_amount", 0)
    tax_type = receipt_data.get("tax_type") or "included"
    savings_advice = receipt_data.get("savings_advice", "")
    items = receipt_data.get("items", [])

    rows_written = 0
    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for item in items:
            # Fallback and quantity coercion
            qty = item.get("quantity", 1)
            try:
                qty = int(qty)
                if qty < 1:
                    qty = 1
            except (ValueError, TypeError):
                qty = 1

            writer.writerow({
                "id": uuid.uuid4().hex,
                "date": date_str,
                "store_name": store_name,
                "japanese_name": item.get("japanese_name", ""),
                "english_name": item.get("english_name", ""),
                "category": item.get("category", "Other"),
                "price": str(item.get("price", 0)),
                "quantity": str(qty),
                "note": item.get("note", ""),
                "receipt_total": str(total_amount),
                "tax_amount": str(tax_amount),
                "tax_type": tax_type,
                "savings_advice": savings_advice,
                "image_path": image_path,
            })
            rows_written += 1

    logger.info(f"Appended {rows_written} item row(s) to CSV ledger for store: {store_name}")
    return rows_written


def load_all_items() -> list:
    """Load every stored item row from the CSV ledger as a list of dicts."""
    init_csv()
    items = []
    with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    return items


def delete_items_from_csv(date_str: str, store_name: str, image_path: str = "") -> bool:
    """Delete all items belonging to a receipt matching the date, store name, and optional image path from the CSV."""
    try:
        records = load_all_items()
        if image_path:
            new_records = [r for r in records if not (r.get("image_path") == image_path)]
        else:
            new_records = [r for r in records if not (r.get("date") == date_str and r.get("store_name") == store_name)]
        
        # Write back to CSV
        with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for r in new_records:
                writer.writerow(r)
        logger.info(f"Successfully deleted items matching date={date_str}, store={store_name}, image_path={image_path} from CSV.")
        return True
    except Exception as e:
        logger.error(f"Error deleting CSV records: {e}")
        return False


def delete_single_item_from_csv(date_str: str, store_name: str, jp_name: str, eng_name: str, item_id: str = "") -> bool:
    """Delete a single line item matching ID or date/store/names from CSV, and update receipt totals."""
    try:
        from main import safe_int  # Safe local import
    except ImportError:
        def safe_int(val):
            try:
                return int(float(val))
            except:
                return 0

    try:
        records = load_all_items()
        
        # Find the target item to delete
        target_item = None
        if item_id:
            for r in records:
                if r.get("id") == item_id:
                    target_item = r
                    break
        
        if not target_item:
            # Fallback to matching names and date/store
            for r in records:
                if (r.get("date") == date_str and 
                    r.get("store_name") == store_name and 
                    r.get("japanese_name") == jp_name and 
                    r.get("english_name") == eng_name):
                    target_item = r
                    break
                    
        if not target_item:
            logger.warning(f"Item not found in CSV. item_id={item_id}, jp_name={jp_name}, eng_name={eng_name}")
            return False
            
        target_image = target_item.get("image_path") or ""
        target_date = target_item.get("date") or ""
        target_store = target_item.get("store_name") or ""
        target_id_to_exclude = target_item.get("id")
        
        target_transaction_items = []
        other_items = []
        
        for r in records:
            if r.get("id") == target_id_to_exclude:
                continue
                
            # Determine if this row belongs to the same receipt
            is_same_receipt = False
            if target_image and r.get("image_path") == target_image:
                is_same_receipt = True
            elif not target_image and r.get("date") == target_date and r.get("store_name") == target_store:
                is_same_receipt = True
                
            if is_same_receipt:
                target_transaction_items.append(r)
            else:
                other_items.append(r)
                
        # Calculate new total amount for this receipt
        tax_amount = 0
        tax_type = "included"
        if target_transaction_items:
            tax_amount = safe_int(target_transaction_items[0].get("tax_amount") or 0)
            tax_type = target_transaction_items[0].get("tax_type") or "included"
            
        subtotal = 0
        discount = 0
        for item in target_transaction_items:
            cat_lower = (item.get("category") or "").lower()
            price_val = safe_int(item.get("price") or 0)
            qty_val = safe_int(item.get("quantity") or 1)
            
            if cat_lower == "change":
                continue
            elif cat_lower == "discount":
                discount += price_val * qty_val
            else:
                subtotal += price_val * qty_val
                
        new_total = subtotal - discount
        if tax_type == "excluded":
            new_total += tax_amount
            
        # Update receipt_total for all remaining items in this transaction
        for item in target_transaction_items:
            item["receipt_total"] = str(new_total)
            
        # Re-assemble records
        new_records = other_items + target_transaction_items
        
        # Write back to CSV
        with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for r in new_records:
                writer.writerow(r)
                
        logger.info(f"Deleted single item '{target_item.get('japanese_name') or target_item.get('english_name')}' from CSV. New total: {new_total}")
        return True
    except Exception as e:
        logger.error(f"Error deleting single item from CSV: {e}")
        return False


def delete_items_by_id(ids) -> int:
    """
    Remove the row(s) whose "id" is in `ids` from the CSV ledger and rewrite the file.
    This is how deletions made in the main window list are persisted.

    Returns the number of rows actually deleted.
    """
    ids_to_delete = set(ids)
    if not ids_to_delete:
        return 0

    init_csv()
    all_items = load_all_items()
    remaining = [row for row in all_items if row.get("id") not in ids_to_delete]
    deleted_count = len(all_items) - len(remaining)

    if deleted_count:
        _write_rows(remaining)
        logger.info(f"Deleted {deleted_count} item row(s) from CSV ledger.")

    return deleted_count


def heal_receipt_totals():
    """Recalculate and correct receipt_total for all transactions in SQLite and CSV, preserving service charges."""
    import json
    try:
        from main import safe_int
    except ImportError:
        def safe_int(val):
            try:
                return int(float(val))
            except:
                return 0

    # 1. Build receipt stats mapping from SQLite raw_ocr_or_json
    receipt_stats = {}
    try:
        from database.db_manager import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT image_file_path, raw_ocr_or_json, tax_amount, tax_type FROM receipts")
        rows = cursor.fetchall()
        for img_path, raw_json_str, db_tax, db_tax_type in rows:
            if not raw_json_str:
                continue
            try:
                raw_json = json.loads(raw_json_str)
                original_total = safe_int(raw_json.get("total_amount") or 0)
                original_tax = safe_int(raw_json.get("tax_amount") or 0)
                original_tax_type = raw_json.get("tax_type") or "included"
                
                # Calculate original subtotal
                original_subtotal = 0
                original_discount = 0
                for item in raw_json.get("items", []) or []:
                    cat_lower = (item.get("category") or "").lower()
                    price_val = safe_int(item.get("price") or 0)
                    qty_val = safe_int(item.get("quantity") or 1)
                    if cat_lower == "change":
                        continue
                    elif cat_lower == "discount":
                        original_discount += price_val * qty_val
                    else:
                        original_subtotal += price_val * qty_val
                
                original_expected = original_subtotal - original_discount
                if original_tax_type == "excluded":
                    original_expected += original_tax
                    
                service_charge_diff = max(0, original_total - original_expected)
                
                service_charge_rate = 0.0
                if original_subtotal > 0:
                    service_charge_rate = service_charge_diff / original_subtotal
                    
                tax_rate = 0.0
                if original_subtotal > 0:
                    tax_rate = original_tax / original_subtotal
                    
                receipt_stats[img_path] = {
                    "original_total": original_total,
                    "original_tax": original_tax,
                    "original_tax_type": original_tax_type,
                    "service_charge_rate": service_charge_rate,
                    "tax_rate": tax_rate
                }
            except Exception as parse_err:
                logger.error(f"Error parsing raw JSON for healing: {parse_err}")
        conn.close()
    except Exception as db_err:
        logger.error(f"Error reading SQLite for healing: {db_err}")

    # 2. Heal CSV
    try:
        from database.csv_manager import load_all_items, CSV_FILE, FIELDNAMES
        import csv
        records = load_all_items()
        if records:
            # Group records by (image_path, date, store_name)
            receipt_groups = {}
            for r in records:
                img = r.get("image_path") or ""
                dt = r.get("date") or ""
                store = r.get("store_name") or ""
                group_key = (img, dt, store)
                
                if group_key not in receipt_groups:
                    receipt_groups[group_key] = []
                receipt_groups[group_key].append(r)
                
            # Recalculate total for each group
            for group_key, items in receipt_groups.items():
                img_path, dt, store = group_key
                stats = receipt_stats.get(img_path)
                
                tax_amount = safe_int(items[0].get("tax_amount") or 0)
                tax_type = items[0].get("tax_type") or "included"
                
                subtotal = 0
                discount = 0
                for item in items:
                    cat_lower = (item.get("category") or "").lower()
                    price_val = safe_int(item.get("price") or 0)
                    qty_val = safe_int(item.get("quantity") or 1)
                    
                    if cat_lower == "change":
                        continue
                    elif cat_lower == "discount":
                        discount += price_val * qty_val
                    else:
                        subtotal += price_val * qty_val
                
                if stats:
                    # Apply rates from original receipt
                    new_tax = int(round(subtotal * stats["tax_rate"]))
                    new_service_charge = int(round(subtotal * stats["service_charge_rate"]))
                    new_total = subtotal - discount + new_service_charge
                    if stats["original_tax_type"] == "excluded":
                        new_total += new_tax
                else:
                    # Fallback to simple calculation
                    new_total = subtotal - discount
                    if tax_type == "excluded":
                        new_total += tax_amount
                        
                # Update all items in this group
                for item in items:
                    item["receipt_total"] = str(new_total)
                    if stats:
                        item["tax_amount"] = str(new_tax)
                        item["tax_type"] = stats["original_tax_type"]
                    
            # Re-assemble records
            new_records = []
            for group_key in receipt_groups:
                new_records.extend(receipt_groups[group_key])
                
            # Write back to CSV
            with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                for r in new_records:
                    writer.writerow(r)
            logger.info("CSV receipt totals successfully healed.")
    except Exception as csv_err:
        logger.error(f"Error healing CSV receipt totals: {csv_err}")
        
    # 3. Heal SQLite
    try:
        from database.db_manager import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all receipts
        cursor.execute("SELECT id, image_file_path, tax_amount, tax_type FROM receipts")
        receipts = cursor.fetchall()
        
        for receipt_id, img_path, db_tax, db_tax_type in receipts:
            stats = receipt_stats.get(img_path)
            
            cursor.execute("SELECT price, quantity, category FROM line_items WHERE receipt_id = ?", (receipt_id,))
            line_items = cursor.fetchall()
            
            subtotal = 0
            discount = 0
            for price, quantity, category in line_items:
                cat_lower = (category or "").lower()
                price_val = float(price or 0)
                qty_val = int(quantity or 1)
                
                if cat_lower == "change":
                    continue
                elif cat_lower == "discount":
                    discount += price_val * qty_val
                else:
                    subtotal += price_val * qty_val
                    
            if stats:
                new_tax = int(round(subtotal * stats["tax_rate"]))
                new_service_charge = int(round(subtotal * stats["service_charge_rate"]))
                new_total = subtotal - discount + new_service_charge
                if stats["original_tax_type"] == "excluded":
                    new_total += new_tax
                cursor.execute("UPDATE receipts SET total_amount = ?, tax_amount = ?, tax_type = ? WHERE id = ?", 
                               (new_total, new_tax, stats["original_tax_type"], receipt_id))
            else:
                db_tax = db_tax or 0
                db_tax_type = db_tax_type or "included"
                new_total = subtotal - discount
                if db_tax_type == "excluded":
                    new_total += db_tax
                cursor.execute("UPDATE receipts SET total_amount = ? WHERE id = ?", (new_total, receipt_id))
            
        conn.commit()
        conn.close()
        logger.info("SQLite receipts totals successfully healed.")
    except Exception as db_err:
        logger.error(f"Error healing SQLite receipt totals: {db_err}")
