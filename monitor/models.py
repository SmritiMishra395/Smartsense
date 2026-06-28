"""
Database models for SmartSense.

SensorReading: every reading from a sensor (anomaly or normal)
AnomalyLog: detailed log of detected anomalies with agent diagnosis
"""
from django.db import models


class SensorReading(models.Model):
    """A single point-in-time sensor reading."""
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    temperature = models.FloatField()
    vibration = models.FloatField()
    power = models.FloatField()
    is_anomaly = models.BooleanField(default=False)
    anomaly_score = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['is_anomaly']),
        ]

    def __str__(self):
        flag = " [ANOMALY]" if self.is_anomaly else ""
        return f"{self.timestamp:%H:%M:%S} | T={self.temperature:.1f} V={self.vibration:.1f} P={self.power:.0f}{flag}"


class AnomalyLog(models.Model):
    """Detailed log of an anomaly, including the full agent diagnosis."""
    SEVERITY_CHOICES = [(i, f"Level {i}") for i in range(1, 6)]

    timestamp = models.DateTimeField(auto_now_add=True)
    reading = models.ForeignKey(
        SensorReading,
        on_delete=models.CASCADE,
        related_name='anomaly_logs'
    )
    anomaly_type = models.CharField(max_length=100)
    severity = models.IntegerField(choices=SEVERITY_CHOICES, default=3)

    # Agent diagnosis fields
    root_cause = models.TextField(blank=True)
    recommended_action = models.TextField(blank=True)
    urgency = models.CharField(max_length=50, blank=True)
    explanation = models.TextField(blank=True)
    confidence = models.CharField(max_length=20, blank=True)
    is_recurring = models.BooleanField(default=False)
    full_diagnosis = models.JSONField(null=True, blank=True)

    # Status
    diagnosis_pending = models.BooleanField(default=True)
    agent_iterations = models.IntegerField(default=0)
    is_resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['severity']),
        ]

    def __str__(self):
        return f"Anomaly #{self.id} | Sev {self.severity} | {self.anomaly_type}"

    @property
    def severity_color(self):
        """Helper for templates — returns Tailwind/CSS color name based on severity."""
        return {1: 'green', 2: 'green', 3: 'amber', 4: 'red', 5: 'red'}.get(self.severity, 'gray')
