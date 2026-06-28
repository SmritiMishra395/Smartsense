"""
Views for SmartSense.

Pages:
  - dashboard:        main live monitor
  - anomaly_detail:   detailed agent diagnosis for a single anomaly
  - anomaly_list:     paginated history of all anomalies

API:
  - simulate_reading: generates a fake sensor reading, runs ML+agent if anomaly
  - chart_data:       returns last 30 readings as JSON for the dashboard chart
"""
import random
import logging
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.utils import timezone
from django.core.paginator import Paginator
from django.views.decorators.csrf import csrf_exempt

from monitor.models import SensorReading, AnomalyLog
from monitor.ml.detect import detect_anomaly
from monitor.agent.agent import run_agent

logger = logging.getLogger(__name__)


def dashboard(request):
    """Main dashboard page."""
    recent_anomalies = AnomalyLog.objects.select_related('reading')[:10]
    total_anomalies = AnomalyLog.objects.count()
    total_readings = SensorReading.objects.count()
    critical_count = AnomalyLog.objects.filter(severity__gte=4).count()

    return render(request, 'monitor/dashboard.html', {
        'recent_anomalies': recent_anomalies,
        'total_anomalies': total_anomalies,
        'total_readings': total_readings,
        'critical_count': critical_count,
    })


def anomaly_detail(request, anomaly_id):
    """Detailed page showing the full agent diagnosis for one anomaly."""
    anomaly = get_object_or_404(AnomalyLog.objects.select_related('reading'), id=anomaly_id)
    return render(request, 'monitor/anomaly_detail.html', {'anomaly': anomaly})


def anomaly_list(request):
    """Paginated list of all anomalies."""
    qs = AnomalyLog.objects.select_related('reading').all()
    paginator = Paginator(qs, 20)
    page = paginator.get_page(request.GET.get('page', 1))
    return render(request, 'monitor/anomaly_list.html', {'page': page})


@csrf_exempt
def simulate_reading(request):
    """
    API: generate a simulated sensor reading.

    10% of readings are intentional anomalies. The ML model decides whether
    to flag, and if flagged, the agent generates a diagnosis.
    """
    try:
        # Decide if we're injecting an anomaly (10% chance)
        is_injected = random.random() < 0.10

        if is_injected:
            anomaly_type = random.choice(['temp', 'vibration', 'power'])
            if anomaly_type == 'temp':
                temperature = random.uniform(38, 50)
                vibration = random.gauss(2.5, 0.8)
                power = random.gauss(300, 50)
            elif anomaly_type == 'vibration':
                temperature = random.gauss(23, 2)
                vibration = random.uniform(12, 20)
                power = random.gauss(300, 50)
            else:  # power_drop
                temperature = random.gauss(23, 2)
                vibration = random.gauss(2.5, 0.8)
                power = random.uniform(10, 50)
        else:
            temperature = random.gauss(23, 2)
            vibration = random.gauss(2.5, 0.8)
            power = random.gauss(300, 50)

        # Clamp to realistic physical ranges
        temperature = max(0, min(temperature, 60))
        vibration = max(0, min(vibration, 25))
        power = max(0, min(power, 700))

        # Run ML detection
        try:
            result = detect_anomaly(temperature, vibration, power)
        except FileNotFoundError as e:
            return JsonResponse({'error': str(e)}, status=500)

        # Save the reading
        reading = SensorReading.objects.create(
            temperature=temperature,
            vibration=vibration,
            power=power,
            is_anomaly=result['is_anomaly'],
            anomaly_score=result['score']
        )

        anomaly_log_id = None

        if result['is_anomaly']:
            # Determine human-readable anomaly type
            features = result.get('anomaly_features', [])
            if features and 'temperature' in features[0]:
                anomaly_type_label = 'Temperature Anomaly'
            elif features and 'vibration' in features[0]:
                anomaly_type_label = 'Vibration Anomaly'
            elif features and 'power' in features[0]:
                anomaly_type_label = 'Power Anomaly'
            elif len(features) >= 2:
                anomaly_type_label = 'Multi-Sensor Anomaly'
            else:
                anomaly_type_label = 'General Anomaly'

            # Create log entry
            log = AnomalyLog.objects.create(
                reading=reading,
                anomaly_type=anomaly_type_label,
                severity=3,
                diagnosis_pending=True
            )

            # Run agent synchronously (for simplicity — in production, use Celery)
            try:
                agent_result = run_agent(result)
            except Exception as e:
                logger.exception("Agent crashed — using fallback")
                from monitor.agent.agent import _fallback_diagnosis
                agent_result = {
                    "success": False,
                    "diagnosis": _fallback_diagnosis(result, str(e)),
                    "iterations": 0
                }

            diagnosis = agent_result.get('diagnosis') or {}
            log.root_cause = diagnosis.get('root_cause', '')
            log.recommended_action = diagnosis.get('recommended_action', '')
            log.urgency = diagnosis.get('urgency', '')
            log.explanation = diagnosis.get('explanation', '')
            log.confidence = diagnosis.get('confidence', '')
            log.is_recurring = diagnosis.get('is_recurring', False)
            log.severity = diagnosis.get('severity_level', 3)
            log.full_diagnosis = diagnosis
            log.diagnosis_pending = False
            log.agent_iterations = agent_result.get('iterations', 0)
            log.save()

            anomaly_log_id = log.id

        return JsonResponse({
            'timestamp': timezone.now().isoformat(),
            'temperature': round(temperature, 2),
            'vibration': round(vibration, 2),
            'power': round(power, 2),
            'is_anomaly': result['is_anomaly'],
            'anomaly_score': round(result['score'], 4),
            'anomaly_features': result['anomaly_features'],
            'anomaly_log_id': anomaly_log_id,
        })

    except Exception as e:
        logger.exception("simulate_reading failed")
        return JsonResponse({'error': str(e)}, status=500)


def chart_data(request):
    """Return last 30 readings for the live chart."""
    readings = list(SensorReading.objects.order_by('-timestamp')[:30])
    readings.reverse()

    return JsonResponse({
        'labels': [r.timestamp.strftime('%H:%M:%S') for r in readings],
        'temperature': [r.temperature for r in readings],
        'vibration': [r.vibration for r in readings],
        'power': [r.power for r in readings],
        'anomalies': [r.is_anomaly for r in readings],
    })


def recent_anomalies_partial(request):
    """Return the rendered HTML for the recent anomalies sidebar (for live refresh)."""
    recent_anomalies = AnomalyLog.objects.select_related('reading')[:10]
    return render(request, 'monitor/_anomaly_list_partial.html', {
        'recent_anomalies': recent_anomalies
    })


# ──────────── NEW FEATURES ────────────


@csrf_exempt
def mark_resolved(request, anomaly_id):
    """Mark an anomaly as resolved by the operator."""
    anomaly = get_object_or_404(AnomalyLog, id=anomaly_id)
    anomaly.is_resolved = True
    anomaly.resolved_at = timezone.now()
    anomaly.save()
    return JsonResponse({'status': 'resolved', 'anomaly_id': anomaly_id})


def export_csv(request):
    """Export all anomalies as CSV — useful for reporting and further analysis."""
    import csv
    from django.http import HttpResponse

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="smartsense_anomalies.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'ID', 'Timestamp', 'Type', 'Severity', 'Urgency',
        'Root Cause', 'Recommended Action', 'Confidence',
        'Temperature', 'Vibration', 'Power', 'ML Score',
        'Is Recurring', 'Is Resolved'
    ])

    anomalies = AnomalyLog.objects.select_related('reading').all()
    for a in anomalies:
        writer.writerow([
            a.id,
            a.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            a.anomaly_type,
            a.severity,
            a.urgency,
            a.root_cause,
            a.recommended_action,
            a.confidence,
            a.reading.temperature if a.reading else '',
            a.reading.vibration if a.reading else '',
            a.reading.power if a.reading else '',
            a.reading.anomaly_score if a.reading else '',
            a.is_recurring,
            getattr(a, 'is_resolved', False),
        ])

    return response


@csrf_exempt
def trigger_test_anomaly(request):
    """
    Manually trigger a specific type of anomaly for testing / demo.
    Used by the dashboard 'Test Trigger' button.
    """
    anomaly_type = request.GET.get('type', 'temp')

    if anomaly_type == 'temp':
        temperature, vibration, power = 47.5, 2.8, 310.0
    elif anomaly_type == 'vibration':
        temperature, vibration, power = 24.0, 16.5, 280.0
    elif anomaly_type == 'power':
        temperature, vibration, power = 23.5, 3.1, 25.0
    elif anomaly_type == 'multi':
        temperature, vibration, power = 42.0, 14.0, 35.0
    else:
        temperature, vibration, power = 45.0, 2.5, 300.0

    try:
        result = detect_anomaly(temperature, vibration, power)
    except FileNotFoundError as e:
        return JsonResponse({'error': str(e)}, status=500)

    reading = SensorReading.objects.create(
        temperature=temperature,
        vibration=vibration,
        power=power,
        is_anomaly=result['is_anomaly'],
        anomaly_score=result['score']
    )

    log = None
    if result['is_anomaly']:
        features = result.get('anomaly_features', [])
        type_labels = {
            'temp': 'Temperature Anomaly',
            'vibration': 'Vibration Anomaly',
            'power': 'Power Anomaly',
            'multi': 'Multi-Sensor Anomaly'
        }
        log = AnomalyLog.objects.create(
            reading=reading,
            anomaly_type=type_labels.get(anomaly_type, 'Test Anomaly'),
            severity=3,
            diagnosis_pending=True
        )

        try:
            agent_result = run_agent(result)
        except Exception as e:
            logger.exception("Agent crashed — using fallback")
            from monitor.agent.agent import _fallback_diagnosis
            agent_result = {
                "success": False,
                "diagnosis": _fallback_diagnosis(result, str(e)),
                "iterations": 0
            }

        diagnosis = agent_result.get('diagnosis') or {}
        log.root_cause = diagnosis.get('root_cause', '')
        log.recommended_action = diagnosis.get('recommended_action', '')
        log.urgency = diagnosis.get('urgency', '')
        log.explanation = diagnosis.get('explanation', '')
        log.confidence = diagnosis.get('confidence', '')
        log.is_recurring = diagnosis.get('is_recurring', False)
        log.severity = diagnosis.get('severity_level', 3)
        log.full_diagnosis = diagnosis
        log.diagnosis_pending = False
        log.agent_iterations = agent_result.get('iterations', 0)
        log.save()

    return JsonResponse({
        'triggered': True,
        'type': anomaly_type,
        'is_anomaly': result['is_anomaly'],
        'anomaly_log_id': log.id if log else None,
        'temperature': temperature,
        'vibration': vibration,
        'power': power,
    })
