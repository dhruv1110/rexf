"""SQLite-based storage implementation for experiment metadata."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.interfaces import StorageInterface
from ..core.models import ExperimentData, ExperimentRun


class SQLiteStorage(StorageInterface):
    """SQLite-based storage for experiment metadata."""

    def __init__(self, db_path: Union[str, Path] = "experiments.db"):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Configure SQLite for better Windows compatibility
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT DEFAULT 'running',
                    parameters TEXT NOT NULL,
                    metrics TEXT,
                    results TEXT,
                    artifacts TEXT,
                    git_commit TEXT,
                    git_status TEXT,
                    environment_info TEXT,
                    random_seed INTEGER,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indices for better query performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_experiment_name
                ON experiments(experiment_name)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_start_time
                ON experiments(start_time)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status
                ON experiments(status)
            """
            )

    def save_experiment(self, experiment: ExperimentData) -> str:
        """Save experiment data."""
        with sqlite3.connect(self.db_path) as conn:
            # Convert complex objects to JSON, handling non-serializable objects
            def safe_json_dumps(obj):
                try:
                    return json.dumps(obj)
                except (TypeError, ValueError):
                    # Convert non-serializable objects to string representation
                    if isinstance(obj, dict):
                        safe_obj = {}
                        for k, v in obj.items():
                            try:
                                json.dumps(v)  # Test if serializable
                                safe_obj[k] = v
                            except (TypeError, ValueError):
                                safe_obj[k] = str(v)
                        return json.dumps(safe_obj)
                    else:
                        return json.dumps(str(obj))

            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    run_id, experiment_name, start_time, end_time, status,
                    parameters, metrics, results, artifacts,
                    git_commit, git_status, environment_info, random_seed, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment.run_id,
                    experiment.experiment_name,
                    experiment.start_time.isoformat(),
                    experiment.end_time.isoformat() if experiment.end_time else None,
                    experiment.status,
                    safe_json_dumps(experiment.parameters),
                    safe_json_dumps(experiment.metrics),
                    safe_json_dumps(experiment.results),
                    safe_json_dumps(experiment.artifacts),
                    experiment.git_commit,
                    safe_json_dumps(experiment.git_status),
                    safe_json_dumps(experiment.environment_info),
                    experiment.random_seed,
                    safe_json_dumps(experiment.metadata),
                ),
            )

        return experiment.run_id

    def load_experiment(self, run_id: str) -> Optional[ExperimentData]:
        """Load experiment by run_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_experiment(row)

    def list_experiments(
        self, experiment_name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[ExperimentData]:
        """List experiments, optionally filtered by name."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM experiments"
            params = []

            if experiment_name:
                query += " WHERE experiment_name = ?"
                params.append(experiment_name)

            query += " ORDER BY start_time DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_experiment(row) for row in rows]

    def update_experiment(self, experiment: ExperimentData) -> None:
        """Update existing experiment."""
        self.save_experiment(experiment)  # INSERT OR REPLACE handles updates

    def delete_experiment(self, run_id: str) -> bool:
        """Delete experiment by run_id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))
            return cursor.rowcount > 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) as total FROM experiments")
            total = cursor.fetchone()[0]

            cursor = conn.execute(
                """
                SELECT experiment_name, COUNT(*) as count
                FROM experiments
                GROUP BY experiment_name
            """
            )
            by_experiment = dict(cursor.fetchall())

            cursor = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM experiments
                GROUP BY status
            """
            )
            by_status = dict(cursor.fetchall())

            # Get database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "total_experiments": total,
                "by_experiment": by_experiment,
                "by_status": by_status,
                "database_size_bytes": db_size,
                "database_path": str(self.db_path),
            }

    def close(self) -> None:
        """Close any open connections and clean up resources."""
        try:
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

    def _row_to_experiment(self, row: sqlite3.Row) -> ExperimentData:
        """Convert database row to ExperimentData."""
        start_time = datetime.fromisoformat(row["start_time"])
        end_time = datetime.fromisoformat(row["end_time"]) if row["end_time"] else None

        experiment = ExperimentRun(
            run_id=row["run_id"],
            experiment_name=row["experiment_name"],
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            start_time=start_time,
            end_time=end_time,
            status=row["status"],
            git_commit=row["git_commit"],
            git_status=json.loads(row["git_status"]) if row["git_status"] else {},
            environment_info=(
                json.loads(row["environment_info"]) if row["environment_info"] else {}
            ),
            random_seed=row["random_seed"],
        )

        experiment.metrics = json.loads(row["metrics"]) if row["metrics"] else {}
        experiment.results = json.loads(row["results"]) if row["results"] else {}
        experiment.artifacts = json.loads(row["artifacts"]) if row["artifacts"] else {}
        experiment.metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return ExperimentData(experiment)
