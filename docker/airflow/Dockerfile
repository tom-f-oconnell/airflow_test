# Airflow image used by every Airflow service (api-server, scheduler, workers).
# Adds the numerical stack the ROI pipeline needs on top of the official image.
ARG AIRFLOW_VERSION=3.3.2
FROM apache/airflow:${AIRFLOW_VERSION}
ARG AIRFLOW_VERSION

COPY docker/airflow/requirements.txt /tmp/requirements.txt
# Pin airflow itself so the extra requirements cannot upgrade/downgrade it.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt

# Baked in so the image is self-contained; docker-compose also bind-mounts the
# working copy over it for development.
COPY --chown=airflow:root roi_pipeline /opt/airflow/roi_pipeline
ENV PYTHONPATH=/opt/airflow
