import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def read_gt():
    gt_path = 'evaluation/ground_truth.json'
    if not os.path.exists(gt_path):
        print("ground_truth.json not found!")
        return
        
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print("All Ground Truth keys:")
    for k in data.keys():
        print(" -", k)
        
    # Search for Yakiniku or Ikuta inside values
    print("\nSearching for Yakiniku/Ikuta inside entries:")
    for k, v in data.items():
        store = v.get("store_name", "")
        if "Ikuta" in store or "Yakiniku" in store or "yakiniku" in store.lower():
            print(f"\nFOUND key '{k}':")
            print("Store:", store)
            print("Total:", v.get("total_amount"))
            print("Tax:", v.get("tax_amount"))
            print("Items:")
            for item in v.get("items", []):
                print(f"  * {item.get('japanese_name')} | {item.get('english_name')} | {item.get('price')} | {item.get('category')}")

if __name__ == "__main__":
    read_gt()
