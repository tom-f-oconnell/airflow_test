# airflow_test

A single-machine test deployment of Apache Airflow 3 for extracting responding units
(ROIs) from volumetric imaging movies by in-memory matrix factorization. The movie lives
on a memory-capped "data node" container, lighter work fans out to compute-worker
containers, and a small website monitors every node, browses the data and launches runs.

## Running it

Requirements: Docker with Compose v2, roughly 10 GB of free disk for images, and a
machine with at least 16 GB of RAM (the default `.env` caps the data node at 8 GB).

```sh
make env      # writes .env (your uid, random Fernet/JWT secrets); edit it to taste
make build    # builds the two images (Airflow + numerical stack, dashboard)
make up       # starts everything; first start also migrates the Airflow DB
```

Then open:

| Site | URL | Login |
|---|---|---|
| Console (nodes, data, launcher) | http://localhost:8090 | none |
| Airflow UI | http://localhost:8080 | airflow / airflow |
| Flower (Celery workers, optional) | http://localhost:5555 after `make flower` | none |

First run, from the console:

1. Open **Generate synthetic movie**, keep the defaults, press *Trigger*. This writes
   `data/datasets/synthetic_small.zarr` on the data node with 12 planted ROIs and a
   stimulus schedule.
2. When the run is green, press **Run factorization** next to the dataset. The run
   plans against the data node's memory cap, factorizes, segments ROIs, classifies them
   and writes `data/results/synthetic_small/<run>/`.
3. Click the dataset name to see the max projection, ROI overlay, dF/F traces, and a
   table with each ROI's quality label, responsive stimuli and response class. Recall
   against the planted ROIs is shown for synthetic data.

Stop with `make down` (keeps the Postgres volume) or `docker compose down -v` to wipe it.

### Setting the data node's memory cap

Edit `DATA_NODE_MEM_LIMIT` in `.env` (Docker sizes: `2g`, `512m`, ...), then
`make restart-data-node`. Only that container is recreated. A one-off override also
works: `DATA_NODE_MEM_LIMIT=2g docker compose up -d data-node`. The console's node table
shows the cap and live usage; the run page shows the cap the `plan` task saw and what it
estimated.

### Using your own data

Drop movies into `data/datasets/` (or point `DATASETS_DIR` in `.env` at them):

- `name.zarr`: a zarr group with array `movie` of shape (T, Z, Y, X); optional attrs
  `stimuli` (see below) and `frame_rate_hz`.
- `name.npy`: (T, Z, Y, X) or (T, Y, X) array; an optional `name.json` sidecar holds attrs.
- `name.tif`: one plane per page, T-major then Z; sidecar as above.

A stimulus schedule enables response classification:
`{"ids": ["A", "B"], "onsets": {"A": [120, 480], "B": [300]}, "duration_frames": 10, "pre_frames": 20}`.

## Configuration

Everything is in `.env` (template: `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `DATA_NODE_MEM_LIMIT` | `8g` | cgroup cap of the data node; swap is capped to the same value |
| `DATA_NODE_CPUS`, `DATA_NODE_THREADS` | 4, 4 | CPU quota and BLAS/OpenMP threads on the data node |
| `DATA_NODE_CONCURRENCY` | 1 | Celery tasks at once on the data node (keep 1 so the plan applies to the whole cap) |
| `COMPUTE_WORKER_REPLICAS` | 2 | number of compute-worker containers |
| `COMPUTE_WORKER_MEM_LIMIT`, `_CPUS`, `_THREADS`, `_CONCURRENCY` | 3g, 2, 2, 2 | per compute worker |
| `AIRFLOW_SERVICE_MEM_LIMIT`, `POSTGRES_MEM_LIMIT`, `REDIS_MEM_LIMIT`, `DASHBOARD_MEM_LIMIT` | 2g, 1g, 512m, 1g | the rest |
| `DATASETS_DIR`, `RESULTS_DIR`, `MODELS_DIR` | `./data/...` | host directories mounted into the nodes |
| `AIRFLOW_PORT`, `DASHBOARD_PORT`, `FLOWER_PORT` | 8080, 8090, 5555 | published ports |
| `AIRFLOW_VERSION` | 3.3.2 | base image tag |

### The `roi_factorization` DAG

Parameters (all editable in the console or Airflow's trigger form):

| Param | Default | Notes |
|---|---|---|
| `dataset` | `synthetic_small` | name under `/data/datasets` |
| `method` | `nmf` | `nmf` or `pca_ica` |
| `n_components` | 20 | rank k |
| `memory_mode` | `auto` | `auto` picks in-memory if the estimate fits the cap, else chunked. `in_memory` forces it even past the cap (expect an OOM kill). `chunked` forces streaming. |
| `spatial_downsample` | 1 | block-mean factor in Y and X before factorization |
| `chunk_frames` | 100 | frames per read, and mini-batch size when chunked |
| `threshold_sigma`, `min_roi_voxels`, `max_rois_per_component` | 3.0, 20, 3 | ROI segmentation |
| `n_response_classes`, `z_threshold` | 3, 2.5 | response classification |

Task placement: `plan`, `factorize_movie` and `extract_rois` run on the data node
(Celery queue `data`); `classify` and `finalize` run on any compute worker (queue
`default`). Outputs per run:

```
data/results/<dataset>/<run>/
  plan.json             cap, estimates, chosen mode, node, params, stimulus schedule
  components.zarr       spatial (k, Z, Y', X') and temporal (T, k) components
  factorization.json    method, timing, peak RSS
  labels.npy            ROI label volume (0 = background)
  traces.npy            raw mean trace per ROI, (T, n_rois)
  rois.json             ROI descriptors
  classification.json   features, quality labels, responsiveness, classes
  summary.json          written last; includes recall on synthetic data
```

### Memory-cap experiments

Generate a movie bigger than the cap, e.g. with `DATA_NODE_MEM_LIMIT=2g`, generate
`n_frames=1500, n_z=16, height=256, width=256` (about 3 GB as uint16, 6 GB as float32).
Then:

- `memory_mode=auto` runs chunked (MiniBatchNMF / IncrementalPCA over time chunks).
- `memory_mode=in_memory` gets OOM-killed. The task fails, the worker survives, and the
  console shows the container with `OOM-killed` on its next restart line.
- `spatial_downsample=2` quarters the matrix and often brings it back under the cap.

### Classification

Two layers run in `classify` (see `roi_pipeline/classify.py`):

- ROI quality (cell / uncertain / artifact) from shape and trace features. Rule-based by
  default; drop a joblib-saved sklearn classifier at `data/models/roi_quality.joblib`
  (train one with `roi_pipeline.classify.train_quality_model` from your hand labels) and it
  is used instead.
- Response type: with a stimulus schedule, a trial-averaged z-score per ROI and stimulus,
  a responsive flag, and k-means over the tuning vectors of the responsive ROIs. Without a
  schedule, k-means over z-scored traces groups co-active ROIs.

`design.md` discusses heavier options (CNN on ROI patches, CaImAn/suite2p classifiers).

### Scaling and monitoring

- More compute workers: `docker compose up -d --scale compute-worker=4`.
- Celery-level view: `make flower`.
- `airflow` CLI: `docker compose --profile debug run --rm airflow-cli dags list`.
- The console reads the Docker socket (read-only) to list containers, their caps and
  live usage. That grants it visibility of every container on the host; fine for a test
  box, not for a shared one.

## Development

```sh
make venv     # ./venv with the pipeline and dev deps
make test
```

The pipeline code is in `roi_pipeline/` and is Airflow-free; the DAGs in `dags/` and
the console in `dashboard/` call into it. Design decisions, the memory-cap mechanics and
the discussion of whether Airflow is the right tool are in [design.md](design.md).

Run the console outside Docker against a running stack with
`PYTHONPATH=. AIRFLOW_API_URL=http://localhost:8080 ROI_DATA_ROOT=./data ./venv/bin/uvicorn dashboard.app:app --reload`.
