"""ROI extraction from volumetric imaging movies by matrix factorization.

The package is deliberately free of Airflow imports so it can be exercised from
the unit tests, the dashboard and plain scripts. The DAGs under ``dags/`` are
thin wrappers that call into it.
"""

__version__ = "0.1.0"
