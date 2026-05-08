# ADMET Predictor

Multi-task ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) prediction API for drug discovery using deep learning.

## Features

- **Multi-task Prediction**: Predict 22 ADMET endpoints simultaneously
- **Deep Learning Models**: Combines graph neural networks (AttentiveFP) with ChemBERTa encoders
- **REST API**: FastAPI-based service for easy integration
- **Batch Processing**: Efficient handling of multiple compounds
- **Explainability**: Attribution methods for model interpretation
- **Uncertainty Estimation**: Evidential deep learning for reliable predictions

## Installation

### Using Poetry (Recommended)

```bash
# Install dependencies
poetry install

# Or with pip
pip install -r requirements.txt
```

### Manual Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torch-geometric rdkit transformers pytorch-lightning mlflow \
    scikit-learn pandas fastapi "uvicorn[standard]" pydantic "redis[asyncio]" celery \
    faiss-cpu captum httpx pytest pytest-asyncio PyTDC molvs pyyaml scipy
```

## Quick Start

### Running the API Server

```bash
# Using Poetry
poetry run uvicorn admet_predictor.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using the Makefile
make serve
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Making Predictions

```bash
curl -X POST "http://localhost:8000/api/v1/admet/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CCO",
    "include_uncertainty": true
  }'
```

## API Endpoints

### Health Check
- `GET /health` - Check if the API is running and model is loaded

### Prediction
- `POST /api/v1/admet/predict` - Predict ADMET properties for a single compound
- `POST /api/v1/admet/batch` - Predict ADMET properties for multiple compounds
- `POST /api/v1/admet/explain` - Get attribution explanations for predictions

### Model Information
- `GET /api/v1/admet/model/info` - Get metadata about the loaded model

## Training

### Download Data

```bash
make download-data
```

### Preprocess Data

```bash
make preprocess
```

### Train Model

```bash
make train
```

## Model Architecture

The ADMET Predictor uses a fusion architecture:

- **Graph Encoder**: AttentiveFP for molecular graph representation
- **Sequence Encoder**: ChemBERTa for SMILES-based representation
- **Task Heads**: Separate prediction heads for each ADMET task
- **Uncertainty**: Evidential deep learning for uncertainty quantification

## ADMET Tasks

The model predicts the following ADMET endpoints:

- **Absorption**: Caco2, HIA, Pgp
- **Distribution**: BBB, PPBR, VDss
- **Metabolism**: CYP2C9, CYP2D6, CYP3A4 substrates/inhibition
- **Excretion**: Clearance (hepatocyte, microsome), Half-life
- **Toxicity**: AMES, hERG, DILI, LD50

## Development

### Running Tests

```bash
# Using Poetry
poetry run pytest tests/ -v

# Or using the Makefile
make test
```

### Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type checking
poetry run mypy src/
```

## Configuration

Model and data configurations are stored in the `configs/` directory:

- `configs/data/admet_tasks.yaml` - Task definitions and data sources
- `configs/model/attentivefp_base.yaml` - Model architecture parameters

## Environment Variables

- `ADMET_CHECKPOINT` - Path to the trained model checkpoint
- `ADMET_DATA_CONFIG` - Path to data configuration file
- `ADMET_MODEL_CONFIG` - Path to model configuration file

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{admet_predictor,
  title = {ADMET Predictor},
  author = {ADMET Team},
  year = {2026},
  url = {https://github.com/AmirhosseinOlyaei/admet-predictor}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- PyTDC for ADMET datasets
- RDKit for cheminformatics
- PyTorch Geometric for graph neural networks
