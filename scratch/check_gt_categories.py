import json
import os

def check_gt_categories():
    gt_path = 'evaluation/ground_truth.json'
    if not os.path.exists(gt_path):
        print("ground_truth.json not found!")
        return
        
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    all_categories = {}
    for filename, receipt in data.items():
        for item in receipt.get("items", []):
            jp = item.get("japanese_name", "")
            eng = item.get("english_name", "")
            cat = item.get("category", "")
            key = (jp, eng, cat)
            all_categories[key] = all_categories.get(key, 0) + 1
            
    print("List of all items and categories in Ground Truth:")
    for (jp, eng, cat), count in sorted(all_categories.items(), key=lambda x: x[0][2]):
        print(f" - Category: {cat:<18} | Jp: {jp:<20} | Eng: {eng:<25} | Count: {count}")

if __name__ == "__main__":
    check_gt_categories()
