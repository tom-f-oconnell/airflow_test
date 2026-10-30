# Design

## What this is for

A one-box rehearsal of a pipeline that turns volumetric imaging movies (T, Z, Y, X) into
ROIs by matrix factorization, classifies the ROIs, and reports on all of it. The point of
the rehearsal is operational: how the pieces behave when the node holding the data has a
hard memory cap, how work is routed to the right node, and what a monitoring surface
needs to show. The numerical methods are deliberately plain so that they are easy to
replace.

## Topology

All nodes are containers from one `docker-compose.yaml`:

| Service | Role | Queue | Mounts |
|---|---|---|---|
| `data-node` | holds datasets, runs the in-memory factorization | `data` | datasets rw, results rw, models ro |
| `compute-worker` ×N | classification, bookkeeping | `default` | results rw, models ro |
| `airflow-apiserver`, `-scheduler`, `-dag-processor`, `-triggerer` | Airflow control plane | | dags, logs |
| `postgres`, `redis` | metadata DB, Celery broker | | |
| `dashboard` | console | | datasets ro, results ro, docker socket ro |

The compute workers cannot see the raw datasets. That is on purpose: it makes the
"data lives on one node" constraint real, so a task that needs the movie must be pinned
to the `data` queue and anything downstream has to work from the small artefacts under
`results/`. On real hardware the same layout maps onto one storage-heavy machine plus
interchangeable workers, with `results/` on shared storage.

Containers rather than VMs because the thing being tested, a cgroup memory cap, is a
container primitive and takes one line. If VMs are wanted later (say, to include the
kernel page cache in the cap), the same compose file runs inside each VM unchanged with
Swarm or plain `docker compose` per VM.

## The memory cap

`DATA_NODE_MEM_LIMIT` becomes `mem_limit` and `memswap_limit` on the data node. The two
are equal, so the cap cannot be dodged by swapping. Inside the container
`roi_pipeline.memory.cgroup_memory_limit` reads `/sys/fs/cgroup/memory.max` (cgroup v2)
or the v1 equivalent, which is what the `plan` task compares against.

Things worth knowing about how the cap bites:

- The cgroup charges page cache for files the container reads. Cache is reclaimable, so
  reading a 20 GB zarr through a 4 GB cap works; it just thrashes the cache.
- Anonymous memory (numpy arrays) is not reclaimable. Exceeding the cap triggers the OOM
  killer inside the cgroup, which kills the largest process. In Airflow 3 the task runs in
  a subprocess supervised by the Celery worker, so the task dies, the worker survives,
  and the task instance is marked failed. The console flags the container as
  `OOM-killed` from Docker's state.
- `DATA_NODE_CONCURRENCY=1` keeps one task on the node at a time so the plan's estimate
  refers to the whole cap. Raise it only with proportionally smaller data.

The estimate in `plan_memory` is intentionally crude: 3× the float32 matrix plus a 512 MB
base for in-memory solvers; chunk-sized terms plus the k×P components for the streaming
ones; 90% of the cap is the budget. It is a guard, not an accounting. `memory_mode`
lets you override it either way, including forcing an in-memory run past the cap to
watch the failure.

## Data flow

```
datasets/<name>.zarr ──(data node)──► plan.json
                                      components.zarr, factorization.json
                                      labels.npy, traces.npy, rois.json
                     ──(compute)────► classification.json
                                      summary.json
```

XCom only carries small dicts (paths, counts, timings). Everything else is a file,
because Airflow's metadata DB is the wrong place for arrays and because files are what
the console reads. `run_dir` names come from the Airflow run id with unsafe characters
replaced, so a run in the Airflow UI and a results directory can be matched.

Zarr is the native format because it is chunked along time: both the synthetic generator
and the chunked solvers stream frames without touching the whole file. `.npy` and
plane-per-page TIFF are accepted for convenience (`roi_pipeline.datasets`).

## Factorization

The movie is the matrix X (T × P), P = Z·Y'·X' after optional block-mean downsampling.

- **NMF**: per-voxel mean subtracted, clipped at zero, `sklearn.decomposition.NMF`
  (nndsvda init). W is T × k temporal, H is k × P spatial. Sparse, positive maps that
  threshold cleanly into cells. Streaming variant: `MiniBatchNMF.partial_fit` over time
  chunks for a couple of epochs, then `transform` per chunk for W.
- **PCA/ICA**: randomized SVD, then FastICA on the k scaled principal maps treated as
  mixtures over voxels (spatial ICA, as in Mukamel et al. 2009), temporal courses by
  least squares. Better at splitting overlapping cells; sign fixed by positive skew.
  Streaming variant: `IncrementalPCA.partial_fit` then the same ICA.

ROI extraction (`roi_pipeline.rois`): each spatial map is robust z-scored, thresholded,
connected components labelled in 3-D, the largest few kept per component, and overlaps
resolved by the stronger z-score. Traces are the mean raw signal over each ROI's voxels,
computed by a sparse averaging matrix so a chunk of frames is all that is ever in memory.

Obvious upgrades once the plumbing is trusted: CNMF (CaImAn) or suite2p's detection for
real cell models with neuropil handling, a motion-correction task in front, and dynamic
task mapping to fan out over datasets or over Z-blocks.

## Classification

Implemented (`roi_pipeline.classify`):

1. **Quality** (cell / uncertain / artifact). Features: voxel count, box compactness,
   aspect ratio, dF/F SNR, skewness, lag-1 autocorrelation, peak dF/F. A rule-based score
   by default; a joblib sklearn model at `models/roi_quality.joblib` if present, trained
   from hand labels with `train_quality_model`. This mirrors suite2p's classifier and is
   the cheapest way to get a lab-specific "is this a cell" filter.
2. **Responsiveness and tuning.** With a stimulus schedule, each ROI's trial-averaged
   response window minus baseline is z-scored against the baseline noise; ROIs with z
   above threshold for any stimulus are responsive, and k-means over their normalised
   tuning vectors gives response classes. Without a schedule, k-means over z-scored traces
   groups co-active ROIs.

Suggested next steps, in rough order of cost:

- **Shape classes from the trial average** (onset, sustained, offset, inhibited): fit a
  few templates to the mean time course per stimulus, or cluster the time courses rather
  than the tuning vectors.
- **Supervised decoder**: train a classifier to predict the stimulus from the population
  response vector per trial (logistic regression or SVM with cross-validation). Decoding
  accuracy is a direct readout of whether the extracted ROIs carry stimulus information,
  which is a better end-to-end check of the factorization than recall on blobs.
- **CNN on ROI patches** (as in CaImAn's component evaluation) for quality, run on a GPU
  worker added as another Celery queue (`gpu`), so it slots into the same DAG.
- **Cross-dataset consistency**: register ROIs across sessions and classify by stability.

## The console

A FastAPI app with server-rendered pages and a little JavaScript:

- **Nodes**: the Docker socket, filtered to the compose project, gives every container's
  state, health, configured cap, live memory (usage minus reclaimable cache, like
  `docker stats`) and CPU. A background thread polls so page loads are instant.
- **Airflow**: health via `/api/v2/monitor/health`; runs, task instances and triggering
  via the REST API with a JWT from `/auth/token`. The launch forms are built from the
  DAG's `Param` schema, so adding a param to a DAG adds a field.
- **Data**: datasets listed from the mount; previews are a max projection over a
  subsample of frames (bounded to ~256 MB of reads) tiled by Z plane; ROI overlays tint
  the label volume; traces are drawn client-side as SVG with a hover crosshair.

Mounting the Docker socket is the one real trade-off: it is the simplest honest source
for cgroup limits and usage, and it is an unacceptable grant on a shared host. The
swap-in for a multi-host deployment is cAdvisor + Prometheus with the console reading
PromQL, which is a one-module change in `dashboard/nodes.py`.

## Is Airflow the right tool?

For what this rehearses, Airflow is a reasonable fit: batch pipelines with a handful of
dependent steps, run many times over many datasets, with retries, history, a UI, and
queue-based placement onto heterogeneous nodes. Those are its strengths. It is not a
fit, and this repo works around it, for:

- **Moving data between steps.** Airflow passes metadata, not arrays. Every step here
  stages through files. If most of the pipeline's complexity is in the array work rather
  than in the step graph, Airflow is mostly overhead.
- **Parallelism inside a step.** The factorization is one task on one node. Airflow does
  not help split an SVD across cores or machines; Dask does.
- **Interactive or low-latency use.** Scheduler latency is seconds; this is not a tool
  for "click and watch the ROIs appear".
- **Weight.** Seven control-plane containers and a Postgres for two DAGs is a lot on one
  box. The `.env` caps keep that to roughly 6 GB, but it is not free.

Alternatives, with when they win:

- **Dagster**: asset-oriented (dataset → components → ROIs → classification as assets
  with lineage and freshness), lighter, better local dev loop, good UI. The most
  natural fit for a scientific pipeline whose outputs are files. If this were going to
  production in a lab rather than being an Airflow exercise, this would be the first
  pick.
- **Prefect**: plain Python flows, minimal infrastructure, decent UI. Easier than Airflow
  for the same DAG, weaker on the multi-queue placement story.
- **Snakemake / Nextflow**: file-driven, HPC-native (SLURM), reproducible, zero
  infrastructure. Best if the lab's compute is a cluster rather than a box of containers,
  and the "node with the data" is a filesystem rather than a machine.
- **Dask (or Ray)**: if the real question is "how do I factorize a movie that does not
  fit in memory", that is an array-computing problem, not a scheduling one. Dask arrays
  over the same zarr give out-of-core SVD and chunked NMF with one worker memory limit
  per process, and the dashboard comes with it. It can also run inside an Airflow task.
- **Celery + a small web app**: what this repo is once you remove the scheduler. Fine for
  "queue a job per dataset, show its status", lacking the history and retry UI.

Recommendation: keep Airflow if the goal includes learning it or integrating with an
organisation that already runs it. Otherwise prototype the same two DAGs in Dagster
(`roi_pipeline/` is untouched by that move) and use Dask inside the factorization step
when the data outgrows one node's memory.

## Not done, on purpose

- No motion correction, neuropil subtraction, or deconvolution.
- No authentication on the console. It is bound to localhost by default.
- Dark mode of the console is not separately validated against the palette gates.
- The stack was validated with `docker compose config`, unit tests of every module, and
  a console smoke test; a full `make up` was not run on the authoring machine because its
  Docker root had under 5 GB free.
