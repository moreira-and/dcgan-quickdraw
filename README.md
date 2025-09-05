# gan-lab-quickdraw

Exploring GANs to generate synthetic images from Google Quick, Draw! (e.g., “coffee cup”). This repository is a lab/portfolio for MLOps and data science, covering data collection and processing, adversarial modeling, sample generation, experiment versioning with MLflow, and CI integration.

---

## Data Source

* [https://quickdraw.withgoogle.com/data/coffee\_cup](https://quickdraw.withgoogle.com/data/coffee_cup)

## Template

* **Badge:** CCDS Template
* [https://cookiecutter-data-science.drivendata.org/](https://cookiecutter-data-science.drivendata.org/)

---

## Lab Goals

* Learn and demonstrate DS/MLOps project best practices.
* Train a simple DCGAN and generate synthetic images (class “coffee cup”).
* Track experiments with MLflow (local and optional Docker server).
* Ensure a minimum quality bar via tests and CI (GitHub Actions).

---

## Project Structure

* `data/`: data at different stages.
* `docs/`: additional documentation (optional mkdocs).
* `mlflow-server/`: Docker environment for MLflow + Postgres + MinIO.
* `mlruns/`: MLflow’s default local store (local tracking URI).
* `models/`: trained artifacts (pkl/pth).
* `notebooks/`: Jupyter notebooks (includes `quick_start.ipynb`).
* `references/`: supporting materials.
* `reports/`: generated outputs and figures.
* `src/`: project source code.
* `tests/`: automated tests (pytest).
* `.github/workflows/ci.yaml`: CI pipeline (installs and runs tests).
* `params.yaml`: hyperparameters (image size, epochs, etc.).
* `Makefile`: automation targets (install, test, lint, format…).
* `pyproject.toml`: project dependencies and config (Poetry/PEP 621).

---

## Requirements

* Python 3.10
* Poetry (environment and dependency management)
* Git
* Docker and Docker Compose (optional, for the MLflow server)
* CUDA GPU (optional; speeds up training)

---

## Installation (Development Environment)

```bash
# 1) Clone the repository
git clone https://github.com/moreira-and/gan-lab-quickdraw.git
cd gan-lab-quickdraw

# 2) Create/select the Poetry environment (Python 3.10)
poetry env use 3.10

# 3) Install dependencies
poetry install

# 4) (Optional) Activate Poetry shell
poetry shell

# 5) Validate installation (tests)
poetry run pytest
```

---

## How to Run `quick_start.ipynb`

The notebook orchestrates the full flow: download data, process features, train the DCGAN, generate a figure, and log artifacts to MLflow (local tracking by default).

```bash
# Launch Jupyter
poetry run jupyter lab
# or
poetry run jupyter notebook
```

Open `notebooks/quick_start.ipynb` and run the cells in order.

---

## Pipeline Steps

* **Load Raw Data:** downloads \~N “coffee cup” drawings from Quick, Draw! and saves them to `data/raw/coffee_cup`.
* **Features:** transforms images into tensors (grayscale, resize) and saves them per class in `data/processed`.
* **Model Training:** trains the DCGAN (hyperparams in `params.yaml`) and saves artifacts to `models/`.
* **Generate:** produces a synthetic image and saves it to `reports/figures/generated_image.png`.
* **Run Experiment:** logs artifacts and parameters to MLflow (local tracking in `mlruns/`).

---

## Expected Outputs

* `data/raw/coffee_cup/`: downloaded PNGs.
* `data/processed/<class>/`: `.pt` tensors per class.
* `models/`: generator/discriminator artifacts.
* `reports/figures/generated_image.png`: generated sample.
* `mlruns/`: MLflow local directory (default tracking store).

---

## Parameters & Configuration

* `params.yaml`: key hyperparameters (e.g., `dataset.batch_size`, `model.generator.latent_dim`, `train.epochs`).
* `src/config.py`: project paths and MLflow default tracking URI at `mlruns/`.

  > Note: by default, tracking is local (`mlruns`). To point to an external server, adjust the tracking URI in this file.

---

## Using MLflow

### Option A — Local (default)

* The code sets `mlflow.set_tracking_uri` to `mlruns/`.

```bash
poetry run mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
# Access http://localhost:5000
```

### Option B — Server with Docker (Postgres + MinIO)

1. Adjust variables in `mlflow-server/.env` if needed.
2. Bring services up:

```bash
cd mlflow-server
docker compose up -d
```

3. Access the UI at `http://localhost:5000`
4. Point the project to the server (edit `src/config.py`):

```python
mlflow.set_tracking_uri("http://localhost:5000")
```

---

## Continuous Integration (CI)

* GitHub Actions pipeline in `.github/workflows/ci.yaml`.
* Triggers: push and pull request to `main`.
* Steps: checkout, setup Python **3.10.5**, install with Poetry, run pytest.

---

## Useful Commands (Makefile)

```bash
make requirements   # install dependencies (poetry install)
make test           # run tests (pytest)
make lint           # style checks (flake8, isort, black --check)
make format         # format code (isort, black)
make data           # download data via src/dataset.py
```

---

## Detailed Folder Structure

* `data/`

  * `data/raw/`: original data (Quick, Draw!).
  * `data/processed/`: transformed data (tensors per class).
  * `data/interim/`, `data/external/`: reserved for other stages/sources.
* `mlflow-server/`

  * `mlflow-server/compose.yaml`: orchestrates Postgres, MinIO, and MLflow.
  * `mlflow-server/.env`: config for credentials/ports/bucket.
* `notebooks/`

  * `notebooks/quick_start.ipynb`: end-to-end project flow.
* `src/`

  * `src/config.py`: sets paths, loads `params.yaml`, defines tracking URI.
  * `src/dataset.py`: downloads and saves Quick, Draw! sketches.
  * `src/features.py`: transforms and saves tensors per class.
  * `src/modeling/`: models (Generator/Discriminator), training, and generation.
  * `src/plots.py`: visualizations from processed data.
* `tests/`: unit tests (e.g., `tests/test_noise.py`).
* `.github/workflows/ci.yaml`: CI pipeline.

---

## Runtime & Performance Notes

* Training: per `params.yaml` (**epochs=100** by default); GPU recommended.
* Environment: project targets **Python >= 3.10** and uses Poetry; the notebook adds `src/` to `sys.path` automatically.

---

## License

* MIT License (see `LICENSE`).

---

## Contact

* This repository is part of my portfolio. Questions or suggestions are welcome via issues/PRs.