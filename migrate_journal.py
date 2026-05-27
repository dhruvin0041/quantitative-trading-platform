import sqlite3
import os

def migrate():
    db_path = "backend/data/empirical_validation.db"
    if not os.path.exists(db_path):
        print("No database found to migrate.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(signals)")
    columns = [col[1] for col in cursor.fetchall()]
    
    expected_columns = [
        ("market_regime_v2", "TEXT"),
        ("quality_score", "REAL"),
        ("quality_grade", "TEXT"),
        ("ev_pct", "REAL"),
        ("expected_gain", "REAL"),
        ("expected_loss", "REAL"),
        ("asset_class", "TEXT")
    ]
    
    for col_name, col_type in expected_columns:
        if col_name not in columns:
            print(f"Adding column {col_name}...")
            cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")
    
    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
