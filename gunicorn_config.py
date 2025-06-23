import os

workers = int(os.environ.get('GUNICORN_WORKERS', 4))
threads = int(os.environ.get('GUNICORN_THREADS', 2))
timeout = int(os.environ.get('GUNICORN_TIMEOUT', 120))
bind = f"0.0.0.0:{os.environ.get('PORT', 8050)}"
worker_class = "gevent"
