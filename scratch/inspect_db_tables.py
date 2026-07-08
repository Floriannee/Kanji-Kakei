import sqlite3

def check_tables():
    conn = sqlite3.connect('database/kakei.db')
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = c.fetchall()
    print("Tables:", tables)
    
    for t in tables:
        t_name = t[0]
        print(f"\nSchema of {t_name}:")
        c.execute(f"PRAGMA table_info({t_name})")
        for col in c.fetchall():
            print(" -", col)

if __name__ == "__main__":
    check_tables()
