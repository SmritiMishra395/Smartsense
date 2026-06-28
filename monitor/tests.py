"""
Tests for SmartSense — Smart Home HVAC Anomaly Detection Agent

Run with:  python manage.py test monitor
"""
from django.test import TestCase, Client as DjangoClient
from django.urls import reverse
from monitor.models import SensorReading, AnomalyLog
from monitor.ml.detect import detect_anomaly
from monitor.agent.agent import _fallback_diagnosis


class MLDetectionTests(TestCase):
    """Test the ML anomaly detection layer."""

    def test_normal_reading_not_flagged(self):
        """Normal HVAC readings should not be flagged as anomalies."""
        result = detect_anomaly(temperature=23.0, vibration=2.5, power=300.0)
        self.assertFalse(result['is_anomaly'])
        self.assertIsInstance(result['score'], float)
        self.assertEqual(result['anomaly_features'], [])

    def test_high_temperature_flagged(self):
        """Temperature spike (e.g., AC compressor failure) should be flagged."""
        result = detect_anomaly(temperature=48.0, vibration=2.5, power=300.0)
        self.assertTrue(result['is_anomaly'])
        self.assertTrue(any('temperature' in f for f in result['anomaly_features']))

    def test_high_vibration_flagged(self):
        """Excessive vibration (e.g., loose compressor mount) should be flagged."""
        result = detect_anomaly(temperature=23.0, vibration=18.0, power=300.0)
        self.assertTrue(result['is_anomaly'])
        self.assertTrue(any('vibration' in f for f in result['anomaly_features']))

    def test_low_power_flagged(self):
        """Power drop (e.g., electrical fault) should be flagged."""
        result = detect_anomaly(temperature=23.0, vibration=2.5, power=25.0)
        self.assertTrue(result['is_anomaly'])
        self.assertTrue(any('power' in f for f in result['anomaly_features']))

    def test_multi_sensor_anomaly(self):
        """Multiple sensors out of range should all be listed."""
        result = detect_anomaly(temperature=45.0, vibration=15.0, power=30.0)
        self.assertTrue(result['is_anomaly'])
        self.assertGreaterEqual(len(result['anomaly_features']), 2)

    def test_detection_returns_required_keys(self):
        """Output dict must always contain the required keys."""
        result = detect_anomaly(temperature=23.0, vibration=2.5, power=300.0)
        for key in ['is_anomaly', 'score', 'anomaly_features', 'readings']:
            self.assertIn(key, result)


class FallbackDiagnosisTests(TestCase):
    """Test the rule-based fallback when the LLM agent is unavailable."""

    def test_critical_severity_for_extreme_score(self):
        """Very negative scores should map to Critical severity."""
        result = _fallback_diagnosis({
            'readings': {'temperature': 50.0, 'vibration': 3.0, 'power': 280},
            'anomaly_features': ['temperature=50.0°C (normal: 18-28°C)'],
            'score': -0.5
        })
        self.assertEqual(result['severity_level'], 5)
        self.assertEqual(result['severity_label'], 'Critical')
        self.assertEqual(result['urgency'], 'Immediate')

    def test_low_severity_for_mild_score(self):
        """Mildly negative scores should map to Low severity."""
        result = _fallback_diagnosis({
            'readings': {'temperature': 30.0, 'vibration': 3.0, 'power': 280},
            'anomaly_features': [],
            'score': -0.03
        })
        self.assertEqual(result['severity_level'], 2)
        self.assertEqual(result['urgency'], 'Monitor only')

    def test_fallback_returns_all_required_fields(self):
        """Fallback must return every field the dashboard expects."""
        result = _fallback_diagnosis({
            'readings': {'temperature': 45.0, 'vibration': 3.0, 'power': 280},
            'anomaly_features': ['temperature=45.0°C (normal: 18-28°C)'],
            'score': -0.3
        })
        for key in ['root_cause', 'severity_level', 'severity_label',
                     'affected_sensors', 'is_recurring', 'recommended_action',
                     'urgency', 'confidence', 'explanation']:
            self.assertIn(key, result)

    def test_affected_sensors_detected(self):
        """Fallback should correctly identify which sensors are anomalous."""
        result = _fallback_diagnosis({
            'readings': {'temperature': 50.0, 'vibration': 18.0, 'power': 280},
            'anomaly_features': [
                'temperature=50.0°C (normal: 18-28°C)',
                'vibration=18.0mm/s (normal: 0-5mm/s)'
            ],
            'score': -0.4
        })
        self.assertIn('temperature', result['affected_sensors'])
        self.assertIn('vibration', result['affected_sensors'])


class ViewTests(TestCase):
    """Test the Django views and API endpoints."""

    def setUp(self):
        self.client = DjangoClient()

    def test_dashboard_loads(self):
        """Dashboard should return 200."""
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'SmartSense')

    def test_anomaly_list_loads(self):
        """Anomaly list page should return 200."""
        response = self.client.get(reverse('anomaly_list'))
        self.assertEqual(response.status_code, 200)

    def test_simulate_endpoint_returns_json(self):
        """Simulate endpoint should return valid JSON with sensor data."""
        response = self.client.get(reverse('simulate'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('temperature', data)
        self.assertIn('vibration', data)
        self.assertIn('power', data)
        self.assertIn('is_anomaly', data)

    def test_chart_data_returns_json(self):
        """Chart data endpoint should return lists for the chart."""
        # Create a few readings first
        for _ in range(5):
            self.client.get(reverse('simulate'))
        response = self.client.get(reverse('chart_data'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('labels', data)
        self.assertIn('temperature', data)
        self.assertIsInstance(data['temperature'], list)

    def test_export_csv(self):
        """CSV export should return a downloadable CSV file."""
        response = self.client.get(reverse('export_csv'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        self.assertIn('smartsense_anomalies.csv', response['Content-Disposition'])

    def test_anomaly_detail_404_for_nonexistent(self):
        """Requesting a non-existent anomaly should return 404."""
        response = self.client.get(reverse('anomaly_detail', args=[99999]))
        self.assertEqual(response.status_code, 404)

    def test_mark_resolved(self):
        """Marking an anomaly as resolved should update the database."""
        # Create a reading and anomaly manually
        reading = SensorReading.objects.create(
            temperature=45.0, vibration=2.5, power=300.0,
            is_anomaly=True, anomaly_score=-0.5
        )
        anomaly = AnomalyLog.objects.create(
            reading=reading,
            anomaly_type='Temperature Anomaly',
            severity=4,
            diagnosis_pending=False
        )
        response = self.client.post(reverse('mark_resolved', args=[anomaly.id]))
        self.assertEqual(response.status_code, 200)

        anomaly.refresh_from_db()
        self.assertTrue(anomaly.is_resolved)
        self.assertIsNotNone(anomaly.resolved_at)


class ModelTests(TestCase):
    """Test the database models."""

    def test_sensor_reading_creation(self):
        """SensorReading should save correctly."""
        reading = SensorReading.objects.create(
            temperature=23.5, vibration=2.1, power=310.0,
            is_anomaly=False
        )
        self.assertIsNotNone(reading.id)
        self.assertIsNotNone(reading.timestamp)

    def test_anomaly_log_creation(self):
        """AnomalyLog should link to SensorReading correctly."""
        reading = SensorReading.objects.create(
            temperature=48.0, vibration=2.5, power=300.0,
            is_anomaly=True, anomaly_score=-0.6
        )
        anomaly = AnomalyLog.objects.create(
            reading=reading,
            anomaly_type='Temperature Anomaly',
            severity=5,
            root_cause='HVAC compressor overheating',
            recommended_action='Shut down unit and inspect',
            urgency='Immediate',
            diagnosis_pending=False
        )
        self.assertEqual(anomaly.reading.temperature, 48.0)
        self.assertEqual(anomaly.severity, 5)
        self.assertFalse(anomaly.is_resolved)

    def test_anomaly_default_values(self):
        """New anomalies should default to unresolved and pending."""
        reading = SensorReading.objects.create(
            temperature=45.0, vibration=2.5, power=300.0, is_anomaly=True
        )
        anomaly = AnomalyLog.objects.create(
            reading=reading, anomaly_type='Test', severity=3
        )
        self.assertTrue(anomaly.diagnosis_pending)
        self.assertFalse(anomaly.is_resolved)
        self.assertIsNone(anomaly.resolved_at)
