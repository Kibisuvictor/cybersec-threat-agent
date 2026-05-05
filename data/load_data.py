# data/load_data.py
import duckdb
import pandas as pd
import os

DB_PATH = "data/cybersec.duckdb"

TABLE_MAP = {
    "otx_threats":        "data/raw/1_otx_threat_intel.csv",
    "cve_vulnerabilities": "data/raw/2_cve_vulnerabilities.csv",
    "malicious_domains":  "data/raw/3_malicious_domains.csv",
    "malicious_ips":      "data/raw/4_malicious_ips.csv",
}

def load_all(db_path: str = DB_PATH):
    os.makedirs("data/raw", exist_ok=True)
    con = duckdb.connect(db_path)

    for table_name, csv_path in TABLE_MAP.items():
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found — skipping {table_name}")
            continue
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{csv_path}', ignore_errors=true)
        """)
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  Loaded {table_name}: {count} rows")

    print("\nAll tables:")
    print(con.execute("SHOW TABLES").fetchdf().to_string(index=False))
    con.close()

if __name__ == "__main__":
    load_all()