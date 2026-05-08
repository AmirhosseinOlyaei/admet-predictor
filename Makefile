.PHONY: install download-data preprocess train test lint serve

PYTHON = .venv/bin/python
PY_ENV = PYTHONPATH=src

install:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
	.venv/bin/pip install torch torchvision torch-geometric rdkit transformers pytorch-lightning mlflow \
		scikit-learn pandas fastapi "uvicorn[standard]" pydantic "redis[asyncio]" celery \
		faiss-cpu captum httpx pytest pytest-asyncio PyTDC molvs pyyaml scipy \
		--trusted-host pypi.org --trusted-host files.pythonhosted.org --retries 5

download-data:
	$(PY_ENV) $(PYTHON) scripts/download_data.py --output-dir data/raw

preprocess:
	$(PY_ENV) $(PYTHON) scripts/preprocess_data.py --raw-dir data/raw --output-dir data/processed

train:
	$(PY_ENV) $(PYTHON) scripts/train.py \
		--config configs/model/attentivefp_base.yaml \
		--data-dir data/processed \
		--output-dir outputs/

test:
	$(PY_ENV) .venv/bin/pytest tests/ -v --tb=short

serve:
	$(PY_ENV) .venv/bin/uvicorn admet_predictor.api.main:app --host 0.0.0.0 --port 8000 --reload
