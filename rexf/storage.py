"""SQLite-based storage implementation for experiment metadata.

This module provides a file-system based storage solution using SQLite
for experiment metadata and a separate artifact management system.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .interfaces import ExperimentMetadata, StorageInterface


class SQLiteStorage(StorageInterface):
    """SQLite-based storage for experiment metadata."""

    def __init__(self, db_path: Union[str, Path] = "experiments.db"):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def close(self):
        """Close any open connections and clean up resources."""
        try:
            # First, close any existing connections by forcing a garbage collection
            import gc

            gc.collect()

            # Connect and clean up WAL files
            conn = sqlite3.connect(self.db_path)
            try:
                # Switch back to DELETE mode to clean up WAL files
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute("VACUUM")
            finally:
                conn.close()

            # Force another garbage collection to ensure cleanup
            gc.collect()

            # Try to remove WAL and SHM files if they exist
            wal_file = Path(str(self.db_path) + "-wal")
            shm_file = Path(str(self.db_path) + "-shm")

            for temp_file in [wal_file, shm_file]:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception:
                    pass

        except Exception:
            pass

    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better Windows compatibility
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    git_commit TEXT,
                    environment_info TEXT,
                    random_seed INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES experiments (run_id) ON DELETE CASCADE
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES experiments (run_id) ON DELETE CASCADE
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES experiments (run_id) ON DELETE CASCADE
                )
            """
            )

            # Create indexes for better query performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments (name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_start_time ON experiments (start_time)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics (run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_results_run_id ON results (run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts (run_id)"
            )

            conn.commit()

    def save_experiment(self, metadata: ExperimentMetadata) -> str:
        """Save experiment metadata and return run_id."""
        if not metadata.run_id:
            metadata.run_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            # Insert main experiment record
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    run_id, experiment_id, name, parameters, start_time, end_time,
                    status, git_commit, environment_info, random_seed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata.run_id,
                    metadata.experiment_id,
                    metadata.name,
                    json.dumps(metadata.parameters),
                    metadata.start_time.isoformat(),
                    metadata.end_time.isoformat() if metadata.end_time else None,
                    metadata.status,
                    metadata.git_commit,
                    json.dumps(metadata.environment_info),
                    metadata.random_seed,
                ),
            )

            # Save metrics
            for name, value in metadata.metrics.items():
                try:
                    json_value = json.dumps(value)
                except (TypeError, ValueError):
                    # Skip non-serializable values (they should be artifacts)
                    json_value = json.dumps(str(value))

                conn.execute(
                    """
                    INSERT OR REPLACE INTO metrics (run_id, name, value)
                    VALUES (?, ?, ?)
                """,
                    (metadata.run_id, name, json_value),
                )

            # Save results
            for name, value in metadata.results.items():
                try:
                    json_value = json.dumps(value)
                except (TypeError, ValueError):
                    # Skip non-serializable values (they should be artifacts)
                    json_value = json.dumps(str(value))

                conn.execute(
                    """
                    INSERT OR REPLACE INTO results (run_id, name, value)
                    VALUES (?, ?, ?)
                """,
                    (metadata.run_id, name, json_value),
                )

            # Save artifact paths
            for name, path in metadata.artifacts.items():
                file_size = None
                if Path(path).exists():
                    file_size = Path(path).stat().st_size

                conn.execute(
                    """
                    INSERT OR REPLACE INTO artifacts (run_id, name, file_path, file_size)
                    VALUES (?, ?, ?, ?)
                """,
                    (metadata.run_id, name, path, file_size),
                )

            conn.commit()

        return metadata.run_id

    def load_experiment(self, run_id: str) -> Optional[ExperimentMetadata]:
        """Load experiment metadata by run_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Load main experiment data
            cursor = conn.execute(
                """
                SELECT * FROM experiments WHERE run_id = ?
            """,
                (run_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            # Parse timestamps
            start_time = datetime.fromisoformat(row["start_time"])
            end_time = (
                datetime.fromisoformat(row["end_time"]) if row["end_time"] else None
            )

            metadata = ExperimentMetadata(
                experiment_id=row["experiment_id"],
                run_id=row["run_id"],
                name=row["name"],
                parameters=json.loads(row["parameters"]),
                start_time=start_time,
                end_time=end_time,
                status=row["status"],
                git_commit=row["git_commit"],
                environment_info=(
                    json.loads(row["environment_info"])
                    if row["environment_info"]
                    else {}
                ),
                random_seed=row["random_seed"],
            )

            # Load metrics
            cursor = conn.execute(
                """
                SELECT name, value FROM metrics WHERE run_id = ?
            """,
                (run_id,),
            )
            for metric_row in cursor.fetchall():
                metadata.metrics[metric_row["name"]] = json.loads(metric_row["value"])

            # Load results
            cursor = conn.execute(
                """
                SELECT name, value FROM results WHERE run_id = ?
            """,
                (run_id,),
            )
            for result_row in cursor.fetchall():
                metadata.results[result_row["name"]] = json.loads(result_row["value"])

            # Load artifacts
            cursor = conn.execute(
                """
                SELECT name, file_path FROM artifacts WHERE run_id = ?
            """,
                (run_id,),
            )
            for artifact_row in cursor.fetchall():
                metadata.artifacts[artifact_row["name"]] = artifact_row["file_path"]

            return metadata

    def list_experiments(
        self, experiment_name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[ExperimentMetadata]:
        """List experiments, optionally filtered by name."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT run_id FROM experiments"
            params = []

            if experiment_name:
                query += " WHERE name = ?"
                params.append(experiment_name)

            query += " ORDER BY start_time DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor = conn.execute(query, params)
            run_ids = [row["run_id"] for row in cursor.fetchall()]

        # Load full metadata for each experiment
        experiments = []
        for run_id in run_ids:
            metadata = self.load_experiment(run_id)
            if metadata:
                experiments.append(metadata)

        return experiments

    def update_experiment(self, metadata: ExperimentMetadata) -> None:
        """Update existing experiment metadata."""
        self.save_experiment(metadata)  # INSERT OR REPLACE handles updates

    def delete_experiment(self, run_id: str) -> bool:
        """Delete experiment by run_id. Returns True if deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM experiments WHERE run_id = ?
            """,
                (run_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Count experiments
            cursor = conn.execute("SELECT COUNT(*) as count FROM experiments")
            stats["total_experiments"] = cursor.fetchone()[0]

            # Count by status
            cursor = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM experiments
                GROUP BY status
            """
            )
            stats["by_status"] = {row[0]: row[1] for row in cursor.fetchall()}

            # Count metrics and results
            cursor = conn.execute("SELECT COUNT(*) as count FROM metrics")
            stats["total_metrics"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) as count FROM results")
            stats["total_results"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) as count FROM artifacts")
            stats["total_artifacts"] = cursor.fetchone()[0]

            # Database file size
            stats["db_size_bytes"] = (
                self.db_path.stat().st_size if self.db_path.exists() else 0
            )

            return stats
