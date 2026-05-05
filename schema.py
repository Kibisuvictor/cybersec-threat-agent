# schema.py
import duckdb

DB_PATH = "data/cybersec.duckdb"

def get_schema_context(db_path: str = DB_PATH) -> str:
    """
    Returns a string describing all tables and columns in the database.
    This gets injected into the agent's system prompt.
    """
    con = duckdb.connect(db_path, read_only=True)
    tables = con.execute("SHOW TABLES").fetchdf()

    schema_parts = []
    for table_name in tables["name"].tolist():
        cols = con.execute(f"DESCRIBE {table_name}").fetchdf()
        col_descriptions = ", ".join(
            f"{row['column_name']} ({row['column_type']})"
            for _, row in cols.iterrows()
        )
        # Add a sample row so the agent understands real values
        sample = con.execute(
            f"SELECT * FROM {table_name} LIMIT 1"
        ).fetchdf().to_dict(orient="records")
        schema_parts.append(
            f"Table: {table_name}\n"
            f"Columns: {col_descriptions}\n"
            f"Sample row: {sample[0] if sample else 'empty'}"
        )

    con.close()
    return "\n\n".join(schema_parts)


if __name__ == "__main__":
    print(get_schema_context())