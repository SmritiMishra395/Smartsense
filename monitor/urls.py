"""URL routes for the monitor app."""
from django.urls import path
from monitor import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('simulate/', views.simulate_reading, name='simulate'),
    path('chart-data/', views.chart_data, name='chart_data'),
    path('anomaly/<int:anomaly_id>/', views.anomaly_detail, name='anomaly_detail'),
    path('anomaly/<int:anomaly_id>/resolve/', views.mark_resolved, name='mark_resolved'),
    path('anomalies/', views.anomaly_list, name='anomaly_list'),
    path('anomalies/recent/', views.recent_anomalies_partial, name='recent_anomalies_partial'),
    path('export/csv/', views.export_csv, name='export_csv'),
    path('test-trigger/', views.trigger_test_anomaly, name='test_trigger'),
]
