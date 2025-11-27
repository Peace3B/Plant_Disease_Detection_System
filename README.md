# Plant Disease Detection ML Pipeline

## Video Demo
YouTube Demo Link - https://www.youtube.com/watch?v=WTyqppva0xo


## Project Description

A complete end-to-end Machine Learning pipeline for plant disease classification using Convolutional Neural Networks (CNN). The system allows users to:
- Upload plant leaf images and get disease predictions
- View model performance metrics and data visualizations
- Upload bulk training data and trigger model retraining
- Monitor model uptime and performance
- Handle concurrent requests with load balancing

**Dataset**: PlantVillage Dataset (subset) - Contains images of healthy and diseased plant leaves across multiple categories.
From Kaggle: https://www.kaggle.com/datasets/arjuntejaswi/plant-village/data

## Architecture

```
User → UI (Streamlit) → FastAPI Backend → ML Model → Predictions
                              ↓
                        Model Monitoring
                              ↓
                      Retraining Pipeline
```

## Directory Structure

```
plant_disease_detection/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── locustfile.py
│
├── notebook/
│   └── plant_disease_classification.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   ├── api.py
│   └── app.py
│
├── data/
│   ├── train/
│   │   ├── healthy/
│   │   └── diseased/
│   └── test/
│       ├── healthy/
│       └── diseased/
│
├── models/
│   ├── plant_disease_model.h5
│   └── model_metadata.json
│
└── uploads/
    └── retrain/
```

## Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- Git

### 1. Clone Repository
```bash
git clone https://github.com/Peace3B/Plant_Disease_Detection_System
cd plant_disease_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
# Download PlantVillage dataset subset
python scripts/download_data.py
```

### 4. Train Initial Model
```bash
# Run the Jupyter notebook or
python src/model.py
```

### 5. Run Application Locally
```bash
# Terminal 1 - Start API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Terminal 2 - Start UI
streamlit run src/app.py --server.port 8501

Web UI: Open http://localhost:8000/web/index.html
API Docs: Open http://localhost:8000/docs
```

### 6. Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Scale containers for load testing
docker-compose up --scale api=3


Access the application at http://localhost:8000/web/index.html.
```

## Load Testing with Locust

### Run Load Test
```bash
# Install Locust
pip install locust

# Run Locust
locust -f locustfile.py --host=http://localhost:8000

# Open browser: http://localhost:8089
# Configure: Users=100, Spawn rate=10
```

## Cloud Deployment (AWS)

### Deploy to AWS ECS
```bash
# 1. Build and push Docker image
docker build -t plant-disease-detection .
docker tag plant-disease-detection:latest <aws-account>.dkr.ecr.us-east-1.amazonaws.com/plant-disease
docker push <aws-account>.dkr.ecr.us-east-1.amazonaws.com/plant-disease

# 2. Create ECS Task Definition
aws ecs create-task-definition --cli-input-json file://task-definition.json

# 3. Create ECS Service
aws ecs create-service --cluster ml-cluster --service-name plant-disease-service --task-definition plant-disease


## Features

### 1. Model Prediction
- Upload single plant leaf image
- Get disease classification with confidence score
- View predicted class and probability distribution

### 2. Data Visualizations
- **Class Distribution**: Bar chart showing training data balance
- **Model Performance**: Confusion matrix and accuracy metrics
- **Training History**: Loss and accuracy curves over epochs
- **Prediction Confidence**: Distribution of model confidence scores

### 3. Bulk Data Upload
- Upload multiple images in ZIP format
- Automatic preprocessing and validation
- Store in staging area for retraining

### 4. Model Retraining
- Trigger button to start retraining process
- Uses existing model + newly uploaded data
- Transfer learning approach for faster training
- Automatic model versioning and backup

### 5. Model Monitoring
- Real-time uptime tracking
- Request count and latency metrics
- Model version information
- Last retrain timestamp

## Model Performance

### Current Model Metrics
- **Accuracy**: 94.3%
- **Precision**: 93.8%
- **Recall**: 94.1%
- **F1-Score**: 93.9%

### Model Architecture
```
- Input Layer: (224, 224, 3)
- Conv2D + MaxPooling (×3)
- Flatten
- Dense (256, ReLU)
- Dropout (0.5)
- Dense (128, ReLU)
- Output Layer (Softmax)
```

## Configuration

Edit `config.py` for custom settings:
```python
MODEL_PATH = "models/plant_disease_model.h5"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
```

## API Endpoints

### Prediction
```bash
POST /predict
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@plant_leaf.jpg"
```

### Upload Training Data
```bash
POST /upload-training-data
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/upload-training-data" \
  -F "files=@images.zip"
```

### Trigger Retraining
```bash
POST /retrain

curl -X POST "http://localhost:8000/retrain"
```

### Model Status
```bash
GET /status

curl "http://localhost:8000/status"
```

## Troubleshooting

### Common Issues

**Issue**: Model file not found
```bash
# Solution: Train model first
python src/model.py
```

**Issue**: Out of memory during training
```bash
# Solution: Reduce batch size in config.py
BATCH_SIZE = 16
```

**Issue**: Docker container crashes
```bash
# Solution: Increase memory allocation
docker-compose up --build --scale api=1
```

## Technologies Used

- **ML Framework**: TensorFlow/Keras
- **API**: FastAPI
- **UI**: Streamlit
- **Containerization**: Docker
- **Load Testing**: Locust
- **Cloud**: AWS ECS / Google Cloud Run
- **Monitoring**: Prometheus + Grafana (optional)

## Contributors

Peace Keza - p.keza@alustudent.com

## License

MIT License

## Acknowledgments

- PlantVillage Dataset
- TensorFlow Team
- FastAPI Community