import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def analyze():
    gt_path = 'evaluation/ground_truth.json'
    if not os.path.exists(gt_path):
        print("ground_truth.json not found!")
        return
        
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Total receipts in ground truth: {len(data)}")
    
    category_items = {}
    for filename, receipt in data.items():
        for item in receipt.get("items", []):
            cat = item.get("category", "Other")
            jp = item.get("japanese_name", "")
            eng = item.get("english_name", "")
            price = item.get("price", 0)
            
            if cat not in category_items:
                category_items[cat] = []
            category_items[cat].append((jp, eng, price))
            
    print("\n--- Ground Truth Categories Analysis ---")
    for cat, items in sorted(category_items.items()):
        print(f"\nCategory: '{cat}' (Total items: {len(items)})")
        seen = set()
        count = 0
        for jp, eng, price in items:
            example = f"{jp} | {eng} (¥{price})"
            if example not in seen:
                seen.add(example)
                print(f"  - {example}")
                count += 1
                if count >= 15:
                    break

if __name__ == "__main__":
    analyze()
