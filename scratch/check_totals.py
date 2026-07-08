import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.csv_manager import heal_receipt_totals, load_all_items

def run_heal_and_check():
    print("Calling heal_receipt_totals()...")
    heal_receipt_totals()
    print("Done. Checking Yakiniku records now...")
    
    records = load_all_items()
    yakiniku_records = [r for r in records if "Ikuta" in r.get("store_name", "")]
    print(f"Total Yakiniku records found: {len(yakiniku_records)}")
    
    groups = {}
    for r in yakiniku_records:
        key = (r.get("image_path"), r.get("date"), r.get("store_name"))
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
        
    print(f"Number of groups: {len(groups)}")
    for key, items in groups.items():
        print(f"\nGroup: {key}")
        print(f"Items count: {len(items)}")
        subtotal = 0
        for item in items:
            p = int(item.get("price") or 0)
            q = int(item.get("quantity") or 1)
            subtotal += p * q
        print(f"Computed Subtotal: {subtotal}")
        print(f"New Receipt Total in CSV: {items[0].get('receipt_total')}")
        print(f"Tax Amount: {items[0].get('tax_amount')}, Tax Type: {items[0].get('tax_type')}")

if __name__ == "__main__":
    run_heal_and_check()
