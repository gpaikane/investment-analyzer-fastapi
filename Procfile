web: gunicorn -k uvicorn.workers.UvicornWorker apis:app
worker: celery -A celery_tasks worker --concurrency=2 --loglevel=info --pool=solo