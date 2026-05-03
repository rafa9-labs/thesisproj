---
name: db-migrator
description: Check and run SQLite migrations for the forex.db data store. Validates schema version, applies pending migrations from pipeline/data_migrator.py, verifies table integrity, and reports migration status. Prevents schema drift between code and database.
---

# Skill: /db-migrator

**Trigger:** User types `/db-migrator` or when database schema issues are suspected.

**Objective:** Validate the SQLite database schema matches what the code expects, apply pending migrations, and verify data integrity.

**Protocol:**

1. **Locate the database:**
   - In dev mode: `data/forex.db` (relative to project root).
   - In desktop mode: `%APPDATA%/fx-ml-backtester/forex.db` (set by `FX_DATA_DIR` env var).
   - Check `api/config.py` for the `db_full_path` setting.

2. **Read migration context:**
   - Read `pipeline/data_migrator.py` for migration definitions.
   - Read `pipeline/data_sqlite.py` for schema creation and `DataStore` class.
   - Identify the current schema version from the `schema_version` table (or `_meta` table).

3. **Schema validation:**
   - Connect to the database and list all tables: `SELECT name FROM sqlite_master WHERE type='table'`.
   - Expected tables: `ohlc`, `trades`, `backtest_results`, `jobs`, `schema_version` (and any others defined in `data_sqlite.py`).
   - For each expected table, verify columns match the schema definition.
   - Check for orphaned tables or columns that exist in the schema but not in the database.

4. **Run pending migrations:**
   ```python
   from pipeline.data_sqlite import DataStore
   from pipeline.data_migrator import run_migrations

   store = DataStore("data/forex.db")
   run_migrations(store)  # Apply any pending migrations
   ```

5. **Data integrity checks:**
   - Verify `ohlc` table has data for expected pairs (EURUSD, etc.).
   - Check date ranges: `SELECT MIN(timestamp), MAX(timestamp) FROM ohlc WHERE pair='EURUSD'`.
   - Verify no duplicate timestamps: `SELECT pair, timeframe, timestamp, COUNT(*) FROM ohlc GROUP BY pair, timeframe, timestamp HAVING COUNT(*) > 1`.
   - Check foreign key integrity if applicable.

6. **Output format:**
   ```
   ## Database Migration Report

   **Database:** data/forex.db
   **Schema Version:** 3 (latest: 3)

   | Check | Status | Details |
   |-------|--------|---------|
   | Tables exist | PASS | 5/5 expected tables found |
   | Schema version | PASS | At latest (v3) |
   | Column match | PASS | All columns match definitions |
   | Data present | PASS | EURUSD: 87,600 rows (10 years H1) |
   | No duplicates | PASS | 0 duplicate timestamps |
   | Foreign keys | PASS | No orphaned records |

   **Migrations applied:** 0 (already at latest)

   **Verdict: PASS** - Database is healthy and up-to-date.
   ```

7. **Failure handling:**
   - If schema version is behind: run `run_migrations()` and re-validate.
   - If columns are missing: identify the migration that adds them and apply it.
   - If database file is missing: create it from `DataStore.__init__()` which auto-creates the schema.
   - If data is missing: suggest running `pipeline/data_downloader.py` to fetch from OANDA.
   - If duplicate timestamps found: suggest dedup query `DELETE FROM ohlc WHERE rowid NOT IN (SELECT MIN(rowid) FROM ohlc GROUP BY pair, timeframe, timestamp)`.

8. **Desktop mode migration:**
   - When running in Electron desktop mode, the database is at `process.resourcesPath/forex.db` or `FX_DATA_DIR/forex.db`.
   - Check `electron/utils.ts` `getUserDataDir()` for the correct path.
   - The `run_server.py` entry point handles `FX_DATA_DIR` and `API_DB_PATH` env vars.

**Important:**
- NEVER drop tables in production. Only additive migrations (ADD COLUMN, CREATE TABLE IF NOT EXISTS).
- Always backup the database before destructive operations: `cp data/forex.db data/forex.db.bak`.
- Report the database file size as a health indicator (should be 50-500MB for 10 years of data).