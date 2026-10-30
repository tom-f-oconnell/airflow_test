FROM python:3.12-slim

WORKDIR /app
COPY docker/dashboard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY dashboard /app/dashboard
COPY roi_pipeline /app/roi_pipeline
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8090
CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8090"]
