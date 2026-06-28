"""
Agent tools — Python functions the Gemini agent can call to investigate anomalies.

These tools give the agent access to:
  1. Sensor history (is this part of a pattern?)
  2. Recent anomalies (is this recurring?)
  3. Severity calculation (how serious is this?)

The TOOL_DEFINITIONS list describes each tool to Gemini in the format
required by the Gemini function-calling API.
"""
import logging
from datetime import timedelta
from django.utils import timezone
from django.db.models import Avg, Max, Min

logger = logging.getLogger(__name__)


def query_sensor_history(sensor_type: str, hours: int = 6) -> dict:
    """Fetch recent sensor readings for a specific sensor."""
    # Import here to avoid circular imports at module load time
    from monitor.models import SensorReading

    if sensor_type not in ('temperature', 'vibration', 'power'):
        return {"error": f"Invalid sensor: {sensor_type}"}

    since = timezone.now() - timedelta(hours=hours)
    queryset = SensorReading.objects.filter(timestamp__gte=since)
    count = queryset.count()

    if count == 0:
        return {
            "sensor": sensor_type,
            "hours_analyzed": hours,
            "count": 0,
            "note": "No historical data found in this window — this is one of the first readings."
        }

    # Aggregate stats
    stats = queryset.aggregate(
        avg=Avg(sensor_type),
        max=Max(sensor_type),
        min=Min(sensor_type)
    )

    # Last 5 readings for trend
    last_5 = list(queryset.order_by('-timestamp').values_list(sensor_type, flat=True)[:5])
    last_5.reverse()

    return {
        "sensor": sensor_type,
        "hours_analyzed": hours,
        "count": count,
        "avg": round(stats['avg'], 2),
        "max": round(stats['max'], 2),
        "min": round(stats['min'], 2),
        "last_5_readings": [round(v, 2) for v in last_5]
    }


def get_recent_anomalies(limit: int = 5) -> dict:
    """Get the most recent anomalies so the agent can detect recurring problems."""
    from monitor.models import AnomalyLog

    recent = AnomalyLog.objects.order_by('-timestamp')[:limit]

    return {
        "recent_count": len(recent),
        "anomalies": [
            {
                "timestamp": a.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "type": a.anomaly_type,
                "severity": a.severity,
                "root_cause": a.root_cause[:100] if a.root_cause else "Pending"
            }
            for a in recent
        ]
    }


def calculate_severity(anomaly_score: float, anomaly_features: list) -> dict:
    """Calculate severity level based on ML score and number of affected sensors."""
    # More negative score = more anomalous in Isolation Forest
    if anomaly_score < -0.30:
        level, label = 5, "Critical"
    elif anomaly_score < -0.15:
        level, label = 4, "High"
    elif anomaly_score < -0.05:
        level, label = 3, "Medium"
    elif anomaly_score < 0:
        level, label = 2, "Low"
    else:
        level, label = 1, "Minimal"

    # Multi-sensor anomalies bump severity by 1 (compounding failures are worse)
    if len(anomaly_features) >= 2 and level < 5:
        level += 1
        if level == 5:
            label = "Critical"
        elif level == 4:
            label = "High"
        elif level == 3:
            label = "Medium"

    return {
        "severity_level": level,
        "severity_label": label,
        "affected_sensors_count": len(anomaly_features),
        "anomaly_score": round(anomaly_score, 4),
        "reasoning": f"Score {anomaly_score:.4f} maps to base level. {len(anomaly_features)} sensor(s) out of range."
    }


def check_maintenance_schedule(device_id: str = "HVAC-01") -> dict:
    """
    Check if the HVAC equipment is due for maintenance.
    Returns the last maintenance date and next scheduled date.
    Helps the agent determine if the anomaly is maintenance-related.
    """
    from django.utils import timezone
    from datetime import timedelta

    # Simulated maintenance schedule (in production, this would query a real system)
    now = timezone.now()
    last_maintenance = now - timedelta(days=45)
    next_scheduled = now + timedelta(days=45)
    is_overdue = (now - last_maintenance).days > 90  # overdue if > 90 days since last

    return {
        "device_id": device_id,
        "last_maintenance": last_maintenance.strftime('%Y-%m-%d'),
        "next_scheduled": next_scheduled.strftime('%Y-%m-%d'),
        "days_since_last": (now - last_maintenance).days,
        "days_until_next": (next_scheduled - now).days,
        "is_overdue": is_overdue,
        "maintenance_type": "Preventive — filter replacement, compressor check, refrigerant levels",
        "note": "Regular HVAC maintenance interval is 90 days for commercial units"
    }


# Tool function registry — used by the agent loop to dispatch calls
TOOLS = {
    "query_sensor_history": query_sensor_history,
    "get_recent_anomalies": get_recent_anomalies,
    "calculate_severity": calculate_severity,
    "check_maintenance_schedule": check_maintenance_schedule,
}


# Tool schemas — describes each tool to the Gemini API for function calling
TOOL_DEFINITIONS = [
    {
        "name": "query_sensor_history",
        "description": (
            "Fetch historical sensor readings to identify if this anomaly is part of a "
            "pattern (gradual drift) or a one-off event. Returns aggregate stats and "
            "the last 5 readings for trend analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sensor_type": {
                    "type": "string",
                    "enum": ["temperature", "vibration", "power"],
                    "description": "Which sensor to query"
                },
                "hours": {
                    "type": "integer",
                    "description": "How many hours of history to analyze (default 6)",
                    "default": 6
                }
            },
            "required": ["sensor_type"]
        }
    },
    {
        "name": "get_recent_anomalies",
        "description": (
            "Get the most recent anomalies from the log to check if this is a recurring "
            "problem. Helps identify cascading failures and chronic issues."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent anomalies to return (default 5)",
                    "default": 5
                }
            }
        }
    },
    {
        "name": "calculate_severity",
        "description": (
            "Calculate the severity level (1-5) of this anomaly based on the ML "
            "anomaly score and the number of affected sensors. Multi-sensor "
            "anomalies are weighted higher."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "anomaly_score": {
                    "type": "number",
                    "description": "The anomaly score from the ML model (more negative = more anomalous)"
                },
                "anomaly_features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of sensors showing anomalous readings"
                }
            },
            "required": ["anomaly_score", "anomaly_features"]
        }
    },
    {
        "name": "check_maintenance_schedule",
        "description": (
            "Check the HVAC equipment maintenance schedule. Returns last and next "
            "maintenance dates. Helps determine if the anomaly could be caused by "
            "missed or overdue maintenance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "string",
                    "description": "Equipment ID to check (default HVAC-01)",
                    "default": "HVAC-01"
                }
            }
        }
    }
]
