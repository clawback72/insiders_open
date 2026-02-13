import sqlite3
import pandas as pd
import os

db = 'insider_database.db'

conn = sqlite3.connect(db)

output_dir = 'csv_exports'
os.makedirs(output_dir, exist_ok=True)

# Get tables from db
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)

# dump each table to separate csv
for table_name in tables['name']:
    print(f"Exporting table {table_name}...")
    
    df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
    csv_file_path = os.path.join(output_dir, f"{table_name}.csv")
    df.to_csv(csv_file_path, index=False)
    
    print(f'Table {table_name} exported tp {csv_file_path}.')
    
conn.close

print('All tables have been exported.')