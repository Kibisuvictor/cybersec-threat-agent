"""
tests/test_sql_tool.py

Integration and unit tests for the cybersecurity threat intelligence agent.
Database tests skip gracefully if cybersec.duckdb hasn't been built yet —
safe to run in CI where raw Kaggle CSVs are not present.
"""

import os
import pytest
import duckdb

DB_PATH = "data/cybersec.duckdb"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def con():
    if not os.path.exists(DB_PATH):
        pytest.skip(
            "Database not found — run `python data/load_data.py` first. "
            "Skipping database tests in CI."
        )
    return duckdb.connect(DB_PATH, read_only=True)


# =============================================================================
# Schema tests — do the expected tables and views exist?
# =============================================================================

class TestSchema:
    def test_all_tables_exist(self, con):
        tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
        expected = [
            "otx_threats",
            "cve_vulnerabilities",
            "malicious_domains",
            "malicious_ips",
        ]
        for table in expected:
            assert table in tables, (
                f"Expected table '{table}' not found in database. "
                f"Tables present: {tables}"
            )


# =============================================================================
# Row count tests — did all four CSVs load with expected record counts?
# =============================================================================

class TestRowCounts:
    def test_otx_threats_row_count(self, con):
        count = con.execute("SELECT COUNT(*) FROM otx_threats").fetchone()[0]
        # Dataset documentation states ~2,365 records; allow ±10% for updates
        assert count >= 2000, (
            f"otx_threats has only {count} rows — expected ~2,365"
        )

    def test_cve_vulnerabilities_row_count(self, con):
        count = con.execute("SELECT COUNT(*) FROM cve_vulnerabilities").fetchone()[0]
        assert count >= 1400, (
            f"cve_vulnerabilities has only {count} rows — expected ~1,585"
        )

    def test_malicious_domains_row_count(self, con):
        count = con.execute("SELECT COUNT(*) FROM malicious_domains").fetchone()[0]
        assert count >= 100, (
            f"malicious_domains has only {count} rows — expected ~162"
        )

    def test_malicious_ips_row_count(self, con):
        count = con.execute("SELECT COUNT(*) FROM malicious_ips").fetchone()[0]
        assert count >= 150, (
            f"malicious_ips has only {count} rows — expected ~200"
        )


# =============================================================================
# Data integrity tests — validate key columns and value ranges
# =============================================================================

    def test_no_null_cve_ids(self, con):
        """CVE IDs are the primary identifier — should not be null."""
        nulls = con.execute("""
            SELECT COUNT(*) FROM cve_vulnerabilities
            WHERE cveID IS NULL
        """).fetchone()[0]
        assert nulls == 0, f"Found {nulls} rows with NULL cve_id"

    def test_cve_years_in_expected_range(self, con):
        """Dataset covers 2024–2026 per documentation."""
        result = con.execute("""
            SELECT
                MIN(YEAR(CAST(dateAdded AS DATE))) as min_year,
                MAX(YEAR(CAST(dateAdded AS DATE))) as max_year
            FROM cve_vulnerabilities
            WHERE dateAdded IS NOT NULL
            AND dateAdded != 'Unknown'
        """).fetchdf()
        if len(result) > 0 and result["min_year"][0] is not None:
            assert result["min_year"][0] >= 2020, (
                "CVE dates go further back than expected"
            )
            assert result["max_year"][0] <= 2027, (
                "CVE dates go further forward than expected"
            )


# =============================================================================
# SQL tool tests — test the tool function directly (no Ollama needed)
# =============================================================================

class TestSQLTool:
    def test_tool_returns_string(self, con):
        from agent import execute_sql
        result = execute_sql.invoke(
            "SELECT cveID, dateAdded FROM cve_vulnerabilities LIMIT 3"
        )
        assert isinstance(result, str), "execute_sql should always return a string"

    def test_tool_returns_markdown_table(self, con):
        from agent import execute_sql
        result = execute_sql.invoke(
            "SELECT cveID, dateAdded FROM cve_vulnerabilities LIMIT 5"
        )
        # Markdown tables contain pipe characters
        assert "|" in result, (
            "execute_sql should return a markdown table with pipe characters"
        )

    def test_tool_handles_bad_sql_gracefully(self):
        """Tool must not raise an exception on invalid SQL."""
        from agent import execute_sql
        result = execute_sql.invoke(
            "SELECT * FROM table_that_does_not_exist_xyz"
        )
        assert isinstance(result, str)
        assert "SQL error" in result, (
            "Bad SQL should return an error string, not raise an exception"
        )

    def test_tool_handles_empty_result(self, con):
        from agent import execute_sql
        result = execute_sql.invoke(
            "SELECT * FROM malicious_ips WHERE 1=0"
        )
        assert isinstance(result, str)
        assert "no results" in result.lower() or "|" in result, (
            "Empty query result should return a meaningful string"
        )

    def test_tool_respects_limit(self, con):
        """Verify the tool doesn't silently return more rows than requested."""
        from agent import execute_sql
        result = execute_sql.invoke(
            "SELECT cveID FROM cve_vulnerabilities LIMIT 5"
        )
        # Count data rows in markdown table (exclude header and separator)
        lines = [l for l in result.strip().split("\n")
                 if l.strip().startswith("|") and "---" not in l]
        # First line is header — data rows are the rest
        data_rows = len(lines) - 1 if len(lines) > 0 else 0
        assert data_rows <= 5, (
            f"Expected at most 5 rows, got {data_rows}"
        )

    def test_case_insensitive_query(self, con):
        """Verify ILIKE works in DuckDB for string matching."""
        from agent import execute_sql
        result = execute_sql.invoke(
            "SELECT COUNT(*) as count FROM cve_vulnerabilities "
            "WHERE vendorProject ILIKE '%microsoft%'"
        )
        assert isinstance(result, str)
        assert "SQL error" not in result, (
            "ILIKE query failed — check DuckDB version supports ILIKE"
        )


# =============================================================================
# Schema introspection tests — verify schema.py output is usable
# =============================================================================

class TestSchemaIntrospection:
    def test_schema_context_is_non_empty(self):
        from schema import get_schema_context
        if not os.path.exists(DB_PATH):
            pytest.skip("Database not found")
        schema = get_schema_context()
        assert isinstance(schema, str)
        assert len(schema) > 100, "Schema context is suspiciously short"

    def test_schema_contains_all_table_names(self):
        from schema import get_schema_context
        if not os.path.exists(DB_PATH):
            pytest.skip("Database not found")
        schema = get_schema_context()
        for table in ["otx_threats", "cve_vulnerabilities",
                      "malicious_domains", "malicious_ips"]:
            assert table in schema, (
                f"Table '{table}' not mentioned in schema context — "
                "the agent won't know it exists"
            )

    def test_schema_contains_column_names(self):
        from schema import get_schema_context
        if not os.path.exists(DB_PATH):
            pytest.skip("Database not found")
        schema = get_schema_context()
        # These are key columns the agent needs to generate correct SQL
        for col in ["dateAdded", "cve_id"]:
            assert col in schema, (
                f"Column '{col}' not found in schema context — "
                "agent may generate incorrect SQL"
            )