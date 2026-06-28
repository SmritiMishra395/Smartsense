"""Admin panel registration."""
from django.contrib import admin
from monitor.models import SensorReading, AnomalyLog


@admin.register(SensorReading)
class SensorReadingAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'temperature', 'vibration', 'power', 'is_anomaly', 'anomaly_score')
    list_filter = ('is_anomaly',)
    ordering = ('-timestamp',)


@admin.register(AnomalyLog)
class AnomalyLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'timestamp', 'anomaly_type', 'severity', 'urgency', 'diagnosis_pending')
    list_filter = ('severity', 'diagnosis_pending', 'is_recurring')
    search_fields = ('root_cause', 'recommended_action')
    ordering = ('-timestamp',)
