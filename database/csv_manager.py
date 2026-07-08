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


def delete_items_from_csv(date_str: str, store_name: str) -> bool:
    """Delete all items belonging to a receipt matching the date and store name from the CSV."""
    try:
        records = load_all_items()
        new_records = [r for r in records if not (r.get("date") == date_str and r.get("store_name") == store_name)]
        
        # Write back to CSV
        with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for r in new_records:
                writer.writerow(r)
        logger.info(f"Successfully deleted items matching date={date_str}, store={store_name} from CSV.")
        return True
    except Exception as e:
        logger.error(f"Error deleting CSV records: {e}")
        return False


def delete_single_item_from_csv(date_str: str, store_name: str, jp_name: str, eng_name: str) -> bool:
    """Delete a single line item matching date, store, and item names from CSV, and update receipt totals."""
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
        
        # Filter out the deleted item
        # Also, recalculate the receipt total for the remaining items in this transaction
        target_transaction_items = []
        other_items = []
        for r in records:
            if r.get("date") == date_str and r.get("store_name") == store_name:
                # If it matches the item to delete, skip it
                if r.get("japanese_name") == jp_name and r.get("english_name") == eng_name:
                    continue
                target_transaction_items.append(r)
            else:
                other_items.append(r)
                
        # Calculate new total amount for this receipt
        tax_amount = 0
        tax_type = "included"
        if target_transaction_items:
            tax_amount = safe_int(target_transaction_items[0].get("tax_amount") or 0)
            tax_type = target_transaction_items[0].get("tax_type") or "included"
            
        items_sum = sum(safe_int(item.get("price")) for item in target_transaction_items)
        new_total = items_sum
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
        logger.info(f"Deleted single item '{jp_name or eng_name}' from CSV matching date={date_str}, store={store_name}. New total: {new_total}")
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
