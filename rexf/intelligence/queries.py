"""Smart query engine for experiment filtering using expressions.

This module enables users to query experiments using simple
expressions like "accuracy > 0.9 and runtime < 60".
"""

import operator
import re
from typing import Any, Dict, List, Optional, Union

from ..backends.intelligent_storage import IntelligentStorage
from ..core.models import ExperimentData


class SmartQueryEngine:
    """
    Expression-based query engine for experiments.

    Supports queries like:
    - "accuracy > 0.9"
    - "learning_rate between 0.001 and 0.01"
    - "status == 'completed' and runtime < 300"
    - "accuracy > 0.9 and param_lr < 0.1"
    """

    def __init__(self, storage: IntelligentStorage):
        """Initialize with storage backend."""
        self.storage = storage
        self.operators = {
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
            "=": operator.eq,
        }

    def query(
        self, query_str: str, limit: Optional[int] = None
    ) -> List[ExperimentData]:
        """
        Execute query expression against experiments.

        Args:
            query_str: Query expression string
            limit: Maximum number of results

        Returns:
            List of matching experiments
        """
        try:
            # Parse the query into structured filters
            parsed = self._parse_query(query_str)

            # Convert to storage query format
            storage_query = self._convert_to_storage_query(parsed)

            # Execute query
            return self.storage.query_experiments(**storage_query, limit=limit)

        except Exception as e:
            print(f"⚠️ Query parsing failed: {e}")
            print("📋 Returning all experiments")
            return self.storage.list_experiments(limit=limit)

    def _parse_query(self, query_str: str) -> Dict[str, Any]:
        """Parse query expression into structured format."""
        # Normalize the query string
        query_str = query_str.strip().lower()

        # Split by 'and' (support for OR coming later)
        conditions = [cond.strip() for cond in query_str.split(" and ")]

        parsed = {
            "general_conditions": {},
            "parameter_filters": {},
            "metric_filters": {},
        }

        for condition in conditions:
            condition_parsed = self._parse_single_condition(condition)
            if condition_parsed:
                category = condition_parsed["category"]
                field = condition_parsed["field"]
                operator_str = condition_parsed["operator"]
                value = condition_parsed["value"]

                if category == "general":
                    parsed["general_conditions"][field] = value
                elif category == "parameter":
                    if field not in parsed["parameter_filters"]:
                        parsed["parameter_filters"][field] = {}
                    parsed["parameter_filters"][field][operator_str] = value
                elif category == "metric":
                    if field not in parsed["metric_filters"]:
                        parsed["metric_filters"][field] = {}
                    parsed["metric_filters"][field][operator_str] = value

        return parsed

    def _parse_single_condition(self, condition: str) -> Optional[Dict[str, Any]]:
        """Parse a single condition like 'accuracy > 0.9'."""
        # Handle special cases first
        if "between" in condition:
            return self._parse_between_condition(condition)

        # Standard operator patterns
        patterns = [
            r"(\w+)\s*(>=|<=|>|<|==|!=|=)\s*([+-]?\d*\.?\d+)",  # Numeric
            r"(\w+)\s*(==|!=|=)\s*'([^']*)'",  # String with quotes
            r"(\w+)\s*(==|!=|=)\s*([a-zA-Z_]\w*)",  # String without quotes
        ]

        for pattern in patterns:
            match = re.match(pattern, condition.strip())
            if match:
                field = match.group(1)
                op = match.group(2)
                value_str = match.group(3)

                # Convert value to appropriate type
                value = self._convert_value(value_str)

                # Determine category (general, parameter, or metric)
                category = self._categorize_field(field)

                # Convert operator to standard format
                op_normalized = self._normalize_operator(op)

                return {
                    "field": field,
                    "operator": op_normalized,
                    "value": value,
                    "category": category,
                }

        return None

    def _parse_between_condition(self, condition: str) -> Optional[Dict[str, Any]]:
        """Parse 'field between min and max' conditions."""
        match = re.match(
            r"(\w+)\s+between\s+([+-]?\d*\.?\d+)\s+and\s+([+-]?\d*\.?\d+)", condition
        )
        if match:
            field = match.group(1)
            min_val = float(match.group(2))
            max_val = float(match.group(3))

            category = self._categorize_field(field)

            # Return as two conditions (>= min and <= max)
            return {
                "field": field,
                "operator": "between",
                "value": {"min": min_val, "max": max_val},
                "category": category,
            }

        return None

    def _categorize_field(self, field: str) -> str:
        """Categorize field as general, parameter, or metric."""
        # General experiment fields
        general_fields = [
            "status",
            "experiment_name",
            "duration",
            "runtime",
            "start_time",
            "end_time",
            "run_id",
        ]

        if field in general_fields:
            return "general"
        elif field.startswith("param_"):
            return "parameter"
        else:
            # Default to metric for unknown fields
            return "metric"

    def _normalize_operator(self, op: str) -> str:
        """Normalize operator to standard format."""
        op_mapping = {
            "=": "eq",
            "==": "eq",
            "!=": "ne",
            ">": "gt",
            ">=": "gte",
            "<": "lt",
            "<=": "lte",
        }
        return op_mapping.get(op, "eq")

    def _convert_value(self, value_str: str) -> Union[int, float, str, bool]:
        """Convert string value to appropriate Python type."""
        # Remove quotes
        value_str = value_str.strip("'\"")

        # Try boolean
        if value_str.lower() in ["true", "false"]:
            return value_str.lower() == "true"

        # Try integer
        try:
            if "." not in value_str:
                return int(value_str)
        except ValueError:
            pass

        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Return as string
        return value_str

    def _convert_to_storage_query(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parsed query to storage query format."""
        query = {}

        # General conditions
        if parsed["general_conditions"]:
            query["conditions"] = {}
            for field, value in parsed["general_conditions"].items():
                # Map field names to storage schema
                if field in ["runtime", "duration"]:
                    query["conditions"]["duration_seconds"] = value
                else:
                    query["conditions"][field] = value

        # Parameter filters
        if parsed["parameter_filters"]:
            query["parameter_filters"] = {}
            for field, filters in parsed["parameter_filters"].items():
                # Remove 'param_' prefix if present
                clean_field = field.replace("param_", "")

                if "between" in filters:
                    between_val = filters["between"]
                    query["parameter_filters"][clean_field] = {
                        "gte": between_val["min"],
                        "lte": between_val["max"],
                    }
                else:
                    query["parameter_filters"][clean_field] = filters

        # Metric filters
        if parsed["metric_filters"]:
            query["metric_filters"] = {}
            for field, filters in parsed["metric_filters"].items():
                if "between" in filters:
                    between_val = filters["between"]
                    query["metric_filters"][field] = {
                        "gte": between_val["min"],
                        "lte": between_val["max"],
                    }
                else:
                    query["metric_filters"][field] = filters

        return query

    def get_query_suggestions(self, experiment_name: Optional[str] = None) -> List[str]:
        """Get suggested queries based on available data."""
        suggestions = []

        # Get parameter and metric names from storage
        try:
            self.storage.get_storage_stats()

            # Parameter-based suggestions
            param_summary = self.storage.get_parameter_space_summary(experiment_name)
            for param_name, info in param_summary.items():
                if info["type"] in ["int", "float"] and info["min_value"] is not None:
                    suggestions.extend(
                        [
                            f"{param_name} > {info['min_value']}",
                            f"{param_name} < {info['max_value']}",
                            f"{param_name} between {info['min_value']} and {info['max_value']}",
                        ]
                    )

            # Metric-based suggestions (common patterns)
            common_metrics = [
                "accuracy",
                "loss",
                "error",
                "score",
                "f1",
                "precision",
                "recall",
            ]
            for metric in common_metrics:
                suggestions.extend(
                    [
                        f"{metric} > 0.9",
                        f"{metric} > 0.8",
                        f"{metric} < 0.1",
                    ]
                )

            # General suggestions
            suggestions.extend(
                [
                    "status == 'completed'",
                    "status == 'failed'",
                    "duration < 60",
                    "duration > 300",
                ]
            )

        except Exception:
            # Fallback suggestions if storage query fails
            suggestions = [
                "accuracy > 0.9",
                "loss < 0.1",
                "status == 'completed'",
                "duration < 60",
            ]

        return suggestions[:10]  # Limit to top 10

    def explain_query(self, query_str: str) -> str:
        """Explain what a query does in human-readable terms."""
        try:
            parsed = self._parse_query(query_str)
            explanations = []

            # Explain general conditions
            for field, value in parsed.get("general_conditions", {}).items():
                explanations.append(f"experiments where {field} is {value}")

            # Explain parameter filters
            for param, filters in parsed.get("parameter_filters", {}).items():
                for op, value in filters.items():
                    op_text = {
                        "gt": "greater than",
                        "gte": "greater than or equal to",
                        "lt": "less than",
                        "lte": "less than or equal to",
                        "eq": "equal to",
                        "ne": "not equal to",
                    }.get(op, op)
                    explanations.append(f"parameter {param} is {op_text} {value}")

            # Explain metric filters
            for metric, filters in parsed.get("metric_filters", {}).items():
                for op, value in filters.items():
                    op_text = {
                        "gt": "greater than",
                        "gte": "greater than or equal to",
                        "lt": "less than",
                        "lte": "less than or equal to",
                        "eq": "equal to",
                        "ne": "not equal to",
                    }.get(op, op)
                    explanations.append(f"metric {metric} is {op_text} {value}")

            if explanations:
                return "Find " + " AND ".join(explanations)
            else:
                return "Query couldn't be parsed - returns all experiments"

        except Exception:
            return "Query parsing failed - returns all experiments"
