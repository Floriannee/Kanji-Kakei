import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

def print_table():
    html_path = 'outputs/dashboard_kanji_kakei.html'
    html = open(html_path, encoding='utf-8').read()
    
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
    if not tbody_match:
        print("No tbody found!")
        return
        
    tbody_content = tbody_match.group(1)
    
    row_pattern = re.compile(r'<tr.*?>(.*?)</tr>', re.DOTALL)
    rows = row_pattern.findall(tbody_content)
    
    print(f"Total rows in items table: {len(rows)}")
    for idx, row in enumerate(rows):
        cells = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)
        clean_cells = [re.sub(r'<[^>]*>', '', c).strip() for c in cells]
        clean_cells = [c for c in clean_cells if c]
        if clean_cells:
            print(f"Row {idx}: {clean_cells}")

if __name__ == "__main__":
    print_table()
