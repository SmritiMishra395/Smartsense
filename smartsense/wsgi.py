"""WSGI config for SmartSense — used by Railway/Heroku for deployment."""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartsense.settings')
application = get_wsgi_application()
