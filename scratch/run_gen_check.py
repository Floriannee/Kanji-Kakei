import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import generate_html_dashboard
from database.csv_manager import load_all_items

def check():
    records = load_all_items()
    print("Script sees last record in CSV:")
    print(" - Store:", records[-1].get("store_name"))
    print(" - Image path:", records[-1].get("image_path"))
    
    print("\nRunning generate_html_dashboard()...")
    generate_html_dashboard()
    
    html_path = 'outputs/dashboard_kanji_kakei.html'
    html = open(html_path, encoding='utf-8').read()
    
    store_matches = re.findall(r'<span class="label">Store Location</span><span class="value">(.*?)</span>', html)
    date_matches = re.findall(r'<span class="label">Transaction Date</span><span class="value">(.*?)</span>', html)
    image_matches = re.findall(r'src="/receipts/(.*?)"', html)
    
    print("\nResulting generated HTML:")
    print(" - Store:", store_matches[0] if store_matches else "Not found")
    print(" - Date:", date_matches[0] if date_matches else "Not found")
    print(" - Image:", image_matches[0] if image_matches else "Not found")

if __name__ == "__main__":
    check()
