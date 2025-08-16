"""Intelligent storage backend optimized for analytics and smart queries.

This extends the basic SQLite storage with rich metadata, parameter space tracking,
and query capabilities for intelligent experiment management.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.models import ExperimentData, ExperimentRun


class IntelligentStorage:
    """
    Analytics-focused storage with rich metadata and query capabilities.

    Enhanced features:
    - Parameter space tracking
    - Experiment relationships/lineage
    - Performance analytics
    - Smart indexing for fast queries
    - Pattern detection support
    """

    def __init__(self, db_path: Union[str, Path] = "experiments.db"):
        """Initialize intelligent storage."""
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize enhanced database schema."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Configure SQLite for better performance
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB

            # Main experiments table (enhanced)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds REAL,
                    status TEXT DEFAULT 'running',
                    
                    -- Experiment data (JSON)
                    parameters TEXT NOT NULL,
                    metrics TEXT,
                    results TEXT,
                    artifacts TEXT,
                    metadata TEXT,
                    
                    -- Reproducibility info
                    git_commit TEXT,
                    git_status TEXT,
                    environment_info TEXT,
                    random_seed INTEGER,
                    
                    -- Analytics fields
                    optimization_target TEXT,
                    experiment_tags TEXT,  -- JSON array
                    experiment_lineage TEXT,  -- Parent experiment IDs
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Parameter space table for efficient querying
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    parameter_name TEXT NOT NULL,
                    parameter_value TEXT NOT NULL,  -- JSON serialized
                    parameter_type TEXT NOT NULL,   -- int, float, str, bool, etc.
                    numeric_value REAL,             -- For numeric parameters only
                    
                    FOREIGN KEY (run_id) REFERENCES experiments (run_id) ON DELETE CASCADE,
                    UNIQUE(run_id, parameter_name)
                )
            """
            )

            # Metrics table for efficient analytics
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_metadata TEXT,  -- JSON for additional info
                    
                    FOREIGN KEY (run_id) REFERENCES experiments (run_id) ON DELETE CASCADE,
                    UNIQUE(run_id, metric_name)
                )
            """
            )

            # Performance analytics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_performance (
                    run_id TEXT PRIMARY KEY,
                    cpu_usage_percent REAL,
                    memory_usage_mb REAL,
                    disk_io_mb REAL,
                    network_io_mb REAL,
                    start_timestamp INTEGER,
                    end_timestamp INTEGER,
                    
                    FOREIGN KEY (run_id) REFERENCES experiments (run_id) ON DELETE CASCADE
                )
            """
            )

            # Experiment relationships for lineage tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_run_id TEXT NOT NULL,
                    child_run_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,  -- 'derived_from', 'similar_to', etc.
                    relationship_metadata TEXT,       -- JSON
                    
                    FOREIGN KEY (parent_run_id) REFERENCES experiments (run_id) ON DELETE CASCADE,
                    FOREIGN KEY (child_run_id) REFERENCES experiments (run_id) ON DELETE CASCADE,
                    UNIQUE(parent_run_id, child_run_id, relationship_type)
                )
            """
            )

            # Create optimized indexes
            self._create_indexes(conn)

    def _create_indexes(self, conn):
        """Create indexes for fast queries."""
        indexes = [
            # Core experiment indexes
            "CREATE INDEX IF NOT EXISTS idx_experiment_name ON experiments(experiment_name)",
            "CREATE INDEX IF NOT EXISTS idx_start_time ON experiments(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_status ON experiments(status)",
            "CREATE INDEX IF NOT EXISTS idx_duration ON experiments(duration_seconds)",
            "CREATE INDEX IF NOT EXISTS idx_optimization_target ON experiments(optimization_target)",
            # Parameter indexes
            "CREATE INDEX IF NOT EXISTS idx_param_name ON experiment_parameters(parameter_name)",
            "CREATE INDEX IF NOT EXISTS idx_param_value ON experiment_parameters(parameter_value)",
            "CREATE INDEX IF NOT EXISTS idx_param_numeric ON experiment_parameters(numeric_value)",
            "CREATE INDEX IF NOT EXISTS idx_param_type ON experiment_parameters(parameter_type)",
            # Metric indexes
            "CREATE INDEX IF NOT EXISTS idx_metric_name ON experiment_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_metric_value ON experiment_metrics(metric_value)",
            # Composite indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_exp_status_time ON experiments(status, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_param_name_value ON experiment_parameters(parameter_name, numeric_value)",
            "CREATE INDEX IF NOT EXISTS idx_metric_name_value ON experiment_metrics(metric_name, metric_value)",
        ]

        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.OperationalError:
                pass  # Index might already exist

    def save_experiment(self, experiment: ExperimentData) -> str:
        """Save experiment with enhanced metadata extraction."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("BEGIN TRANSACTION")

            try:
                # Calculate duration
                duration = experiment.duration

                # Extract optimization target from metadata
                opt_target = getattr(experiment, "optimization_target", None)
                if hasattr(experiment, "_experiment_metadata"):
                    opt_target = experiment._experiment_metadata.get("optimize_for")

                # Main experiment record
                conn.execute(
                    """
                    INSERT OR REPLACE INTO experiments (
                        run_id, experiment_name, start_time, end_time, duration_seconds,
                        status, parameters, metrics, results, artifacts, metadata,
                        git_commit, git_status, environment_info, random_seed,
                        optimization_target, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        experiment.run_id,
                        experiment.experiment_name,
                        experiment.start_time.isoformat(),
                        (
                            experiment.end_time.isoformat()
                            if experiment.end_time
                            else None
                        ),
                        duration,
                        experiment.status,
                        self._safe_json_dumps(experiment.parameters),
                        self._safe_json_dumps(experiment.metrics),
                        self._safe_json_dumps(experiment.results),
                        self._safe_json_dumps(experiment.artifacts),
                        self._safe_json_dumps(experiment.metadata),
                        experiment.git_commit,
                        self._safe_json_dumps(experiment.git_status),
                        self._safe_json_dumps(experiment.environment_info),
                        experiment.random_seed,
                        opt_target,
                        datetime.now().isoformat(),
                    ),
                )

                # Store parameters in separate table for efficient querying
                self._store_parameters(conn, experiment.run_id, experiment.parameters)

                # Store metrics in separate table for analytics
                self._store_metrics(conn, experiment.run_id, experiment.metrics)

                conn.execute("COMMIT")
                return experiment.run_id

            except Exception as e:
                conn.execute("ROLLBACK")
                raise e

    def _store_parameters(self, conn, run_id: str, parameters: Dict[str, Any]):
        """Store parameters in normalized table."""
        # Clear existing parameters
        conn.execute("DELETE FROM experiment_parameters WHERE run_id = ?", (run_id,))

        for param_name, param_value in parameters.items():
            param_type = type(param_value).__name__
            numeric_value = None

            # Extract numeric value for efficient querying
            if isinstance(param_value, (int, float)) and not isinstance(
                param_value, bool
            ):
                numeric_value = float(param_value)

            conn.execute(
                """
                INSERT INTO experiment_parameters 
                (run_id, parameter_name, parameter_value, parameter_type, numeric_value)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    param_name,
                    json.dumps(param_value),
                    param_type,
                    numeric_value,
                ),
            )

    def _store_metrics(self, conn, run_id: str, metrics: Dict[str, Any]):
        """Store metrics in normalized table."""
        # Clear existing metrics
        conn.execute("DELETE FROM experiment_metrics WHERE run_id = ?", (run_id,))

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and not isinstance(
                metric_value, bool
            ):
                conn.execute(
                    """
                    INSERT INTO experiment_metrics 
                    (run_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """,
                    (run_id, metric_name, float(metric_value)),
                )

    def query_experiments(
        self,
        conditions: Optional[Dict[str, Any]] = None,
        parameter_filters: Optional[Dict[str, Any]] = None,
        metric_filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ExperimentData]:
        """
        Advanced experiment querying with multiple filter types.

        Args:
            conditions: General conditions (status, experiment_name, etc.)
            parameter_filters: Parameter-based filters {"param_name": {"gt": 0.5}}
            metric_filters: Metric-based filters {"accuracy": {"gte": 0.9}}
            order_by: Order by field (e.g., "start_time", "duration_seconds")
            limit: Maximum number of results

        Returns:
            List of matching experiments
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build complex query
            query_parts = ["SELECT DISTINCT e.* FROM experiments e"]
            where_conditions = []
            params = []

            # Add parameter filters
            if parameter_filters:
                query_parts.append(
                    "JOIN experiment_parameters ep ON e.run_id = ep.run_id"
                )
                for param_name, filters in parameter_filters.items():
                    for op, value in filters.items():
                        where_conditions.append(
                            f"(ep.parameter_name = ? AND ep.numeric_value {self._get_sql_operator(op)} ?)"
                        )
                        params.extend([param_name, value])

            # Add metric filters
            if metric_filters:
                query_parts.append("JOIN experiment_metrics em ON e.run_id = em.run_id")
                for metric_name, filters in metric_filters.items():
                    for op, value in filters.items():
                        where_conditions.append(
                            f"(em.metric_name = ? AND em.metric_value {self._get_sql_operator(op)} ?)"
                        )
                        params.extend([metric_name, value])

            # Add general conditions
            if conditions:
                for field, value in conditions.items():
                    if field in ["status", "experiment_name", "optimization_target"]:
                        where_conditions.append(f"e.{field} = ?")
                        params.append(value)

            # Combine query
            if where_conditions:
                query_parts.append("WHERE " + " AND ".join(where_conditions))

            if order_by:
                query_parts.append(f"ORDER BY e.{order_by} DESC")
            else:
                query_parts.append("ORDER BY e.start_time DESC")

            if limit:
                query_parts.append(f"LIMIT {limit}")

            query = " ".join(query_parts)
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_experiment(row) for row in rows]

    def get_parameter_space_summary(
        self, experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of parameter space exploration."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    parameter_name,
                    parameter_type,
                    COUNT(DISTINCT parameter_value) as unique_values,
                    MIN(numeric_value) as min_value,
                    MAX(numeric_value) as max_value,
                    AVG(numeric_value) as avg_value
                FROM experiment_parameters ep
                JOIN experiments e ON ep.run_id = e.run_id
            """
            params = []

            if experiment_name:
                query += " WHERE e.experiment_name = ?"
                params.append(experiment_name)

            query += " GROUP BY parameter_name, parameter_type"

            cursor = conn.execute(query, params)
            results = cursor.fetchall()

            summary = {}
            for row in results:
                summary[row[0]] = {
                    "type": row[1],
                    "unique_values": row[2],
                    "min_value": row[3],
                    "max_value": row[4],
                    "avg_value": row[5],
                }

            return summary

    def get_metric_trends(
        self, metric_name: str, experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get trends for a specific metric over time."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    e.start_time,
                    em.metric_value,
                    e.run_id
                FROM experiment_metrics em
                JOIN experiments e ON em.run_id = e.run_id
                WHERE em.metric_name = ?
            """
            params = [metric_name]

            if experiment_name:
                query += " AND e.experiment_name = ?"
                params.append(experiment_name)

            query += " ORDER BY e.start_time"

            cursor = conn.execute(query, params)
            results = cursor.fetchall()

            if not results:
                return {}

            timestamps = [row[0] for row in results]
            values = [row[1] for row in results]
            run_ids = [row[2] for row in results]

            return {
                "metric_name": metric_name,
                "timestamps": timestamps,
                "values": values,
                "run_ids": run_ids,
                "trend": "improving" if values[-1] > values[0] else "declining",
                "min_value": min(values),
                "max_value": max(values),
                "latest_value": values[-1],
            }

    def _get_sql_operator(self, op: str) -> str:
        """Convert filter operator to SQL."""
        operators = {
            "gt": ">",
            "gte": ">=",
            "lt": "<",
            "lte": "<=",
            "eq": "=",
            "ne": "!=",
        }
        return operators.get(op, "=")

    def _safe_json_dumps(self, obj: Any) -> str:
        """Safe JSON serialization."""
        try:
            return json.dumps(obj)
        except (TypeError, ValueError):
            return json.dumps(str(obj))

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

    # Implement remaining interface methods
    def load_experiment(self, run_id: str) -> Optional[ExperimentData]:
        """Load experiment by run_id."""
        results = self.query_experiments(conditions={"run_id": run_id}, limit=1)
        return results[0] if results else None

    def list_experiments(
        self, experiment_name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[ExperimentData]:
        """List experiments."""
        conditions = {"experiment_name": experiment_name} if experiment_name else None
        return self.query_experiments(conditions=conditions, limit=limit)

    def update_experiment(self, experiment: ExperimentData) -> None:
        """Update experiment."""
        self.save_experiment(experiment)

    def delete_experiment(self, run_id: str) -> bool:
        """Delete experiment and related data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))
            return cursor.rowcount > 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get enhanced storage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}

            # Basic counts
            cursor = conn.execute("SELECT COUNT(*) FROM experiments")
            stats["total_experiments"] = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT experiment_name, COUNT(*) FROM experiments GROUP BY experiment_name"
            )
            stats["by_experiment"] = dict(cursor.fetchall())

            cursor = conn.execute(
                "SELECT status, COUNT(*) FROM experiments GROUP BY status"
            )
            stats["by_status"] = dict(cursor.fetchall())

            # Parameter space stats
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT parameter_name) FROM experiment_parameters"
            )
            stats["unique_parameters"] = cursor.fetchone()[0]

            # Metric stats
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT metric_name) FROM experiment_metrics"
            )
            stats["unique_metrics"] = cursor.fetchone()[0]

            # Performance stats
            cursor = conn.execute(
                "SELECT AVG(duration_seconds), MIN(duration_seconds), MAX(duration_seconds) FROM experiments WHERE duration_seconds IS NOT NULL"
            )
            row = cursor.fetchone()
            if row[0]:
                stats["performance"] = {
                    "avg_duration": row[0],
                    "min_duration": row[1],
                    "max_duration": row[2],
                }

            stats["database_size_bytes"] = (
                self.db_path.stat().st_size if self.db_path.exists() else 0
            )
            stats["database_path"] = str(self.db_path)

            return stats

    def close(self) -> None:
        """Close storage and cleanup."""
        try:
            import gc

            gc.collect()

            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute("VACUUM")
            finally:
                conn.close()

            gc.collect()

            # Clean up WAL files
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
