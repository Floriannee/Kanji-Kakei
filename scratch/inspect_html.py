import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.csv_manager import load_all_items

def inspect():
    records = load_all_items()
    print("Latest record in database:")
    print(" - Store:", records[-1].get("store_name"))
    print(" - Date:", records[-1].get("date"))
    print(" - Image path:", records[-1].get("image_path"))
    
    html_path = 'outputs/dashboard_kanji_kakei.html'
    if not os.path.exists(html_path):
        print("HTML file does not exist!")
        return
        
    html = open(html_path, encoding='utf-8').read()
    
    store_matches = re.findall(r'<span class="label">Store Location</span><span class="value">(.*?)</span>', html)
    date_matches = re.findall(r'<span class="label">Transaction Date</span><span class="value">(.*?)</span>', html)
    image_matches = re.findall(r'src="/receipts/(.*?)"', html)
    
    print("\nLatest Receipt in HTML file:")
    print(" - Store:", store_matches[0] if store_matches else "Not found")
    print(" - Date:", date_matches[0] if date_matches else "Not found")
    print(" - Image URL:", image_matches[0] if image_matches else "Not found")

if __name__ == "__main__":
    inspect()
