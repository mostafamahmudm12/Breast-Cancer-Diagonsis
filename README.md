# 🤖 ML Model Prediction API

A FastAPI-based machine learning inference service with a Streamlit web interface for making predictions using multiple trained models.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Models](#models)
- [Development](#development)
- [Contributing](#contributing)

## ✨ Features

- **FastAPI Backend**: High-performance API with automatic documentation
- **Multiple ML Models**: Support for KNN, Logistic Regression, Random Forest, and SVM
- **Streamlit Frontend**: User-friendly web interface for predictions
- **Authentication**: API key-based security
- **Batch Predictions**: Support for multiple samples in a single request
- **Model Management**: Easy model loading and switching
- **CORS Support**: Cross-origin resource sharing enabled
- **Comprehensive Logging**: Request and error logging

## 📁 Project Structure

```
ml-prediction-api/
│
├── main.py                    # FastAPI application entry point
├── streamapp.py              # Streamlit web interface
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create from .env.example)
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore file
├── README.md                # This file
│
├── src/                     # Source code directory
│   ├── config.py            # Configuration settings
│   ├── inference.py         # Model inference logic
│   ├── schemas.py           # Pydantic data models
│   │
│   ├── assets/              # Trained model files
│   │   ├── KNN_model.pkl
│   │   ├── logistic_regression_model.pkl
│   │   ├── Random_model.pkl
│   │   ├── scaler.pkl
│   │   └── SVM_model.pkl
│   │
│   ├── Data/                # Training data
│   │   └── data.csv
│   │
│   └── notebooks/           # Jupyter notebooks
│       └── Breast.ipynb     # Model training notebook
│
└── __pycache__/             # Python cache files
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ml-prediction-api
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your configuration
nano .env  # or use your preferred editor
```

## ⚙️ Configuration

Create a `.env` file in the project root with the following variables:

```env
# Application Settings
APP_NAME=ML Prediction API
VERSION=1.0.0
API_SECRET_KEY=your-secret-api-key-here

# Server Settings
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Logging
LOG_LEVEL=INFO
```

### Configuration Options

- `APP_NAME`: Name of your application
- `VERSION`: Application version
- `API_SECRET_KEY`: Secret key for API authentication
- `HOST`: Server host address
- `PORT`: Server port number
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## 🔧 Usage

### Running the FastAPI Server

```bash
# Development mode with auto-reload
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Running the Streamlit Interface

```bash
# In a new terminal window
streamlit run streamapp.py
```

The Streamlit app will be available at `http://localhost:8501`

### API Documentation

Once the server is running, you can access:

- **Interactive API docs (Swagger)**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## 📡 API Documentation

### Authentication

All API endpoints require authentication using an API key in the header:

```bash
X-API-Key: your-secret-api-key-here
```

### Endpoints

#### Health Check
```http
GET /
```

**Response:**
```json
{
  "app_name": "ML Prediction API",
  "version": "1.0.0",
  "status": "up & running"
}
```

#### Make Prediction
```http
POST /predict/{model_name}
```

**Parameters:**
- `model_name` (path): Name of the model to use (`KNN`, `logistic_regression`, `Random`, `SVM`)

**Request Body:**
```json
{
  "samples": [
    {
      "feature1": 1.0,
      "feature2": 2.0,
      "feature3": 3.0,
      "feature4": 4.0
    }
  ]
}
```

**Response:**
```json
{
  "model_name": "KNN",
  "predictions": [0],
  "probabilities": [[0.8, 0.2]],
  "num_samples": 1
}
```

### Example Usage with curl

```bash
# Health check
curl -X GET "http://localhost:8000/" \
  -H "X-API-Key: your-secret-api-key-here"

# Make prediction
curl -X POST "http://localhost:8000/predict/KNN" \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "feature1": 1.0,
        "feature2": 2.0,
        "feature3": 3.0,
        "feature4": 4.0
      }
    ]
  }'
```

## 🧠 Models

The application supports the following pre-trained models:

### Available Models

1. **KNN (K-Nearest Neighbors)**
   - File: `src/assets/KNN_model.pkl`
   - Good for: Non-linear patterns, local decision boundaries

2. **Logistic Regression**
   - File: `src/assets/logistic_regression_model.pkl`
   - Good for: Linear relationships, probabilistic outputs

3. **Random Forest**
   - File: `src/assets/Random_model.pkl`
   - Good for: Complex patterns, feature importance

4. **SVM (Support Vector Machine)**
   - File: `src/assets/SVM_model.pkl`
   - Good for: High-dimensional data, robust to outliers

### Data Preprocessing

- **Scaler**: `src/assets/scaler.pkl` - StandardScaler for feature normalization
- All input features are automatically scaled before prediction

### Model Training

The models were trained using the data in `src/Data/data.csv`. You can find the training process in the Jupyter notebook `src/notebooks/Breast.ipynb`.

## 🛠️ Development

### Adding New Models

1. Train your model and save it as a pickle file in `src/assets/`
2. Update the model loading logic in `src/inference.py`
3. Add the model name to the available models list

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest

# Run tests with coverage
pytest --cov=src
```

### Code Quality

```bash
# Install development dependencies
pip install black flake8 isort

# Format code
black .

# Sort imports
isort .

# Check code style
flake8 .
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t ml-prediction-api .
```

### Run Container

```bash
docker run -d \
  --name ml-api \
  -p 8000:8000 \
  -e API_SECRET_KEY=your-secret-key \
  ml-prediction-api
```

## 🔍 Monitoring and Logging

The application includes comprehensive logging:

- Request logging with timestamps
- Error tracking and debugging
- Performance monitoring
- API access logging

Logs are written to the console and can be redirected to files:

```bash
python main.py > app.log 2>&1
```

## 🚨 Error Handling

The API includes robust error handling:

- **400 Bad Request**: Invalid input data
- **403 Forbidden**: Invalid API key
- **404 Not Found**: Model not found
- **500 Internal Server Error**: Server-side errors

## 🔐 Security

- API key authentication on all endpoints
- Input validation using Pydantic models
- CORS configuration for cross-origin requests
- Request rate limiting (can be configured)

## 📊 Performance

- Async FastAPI for high performance
- Efficient model loading and caching
- Batch prediction support
- Optimized inference pipeline

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [API documentation](#api-documentation)
2. Review the [troubleshooting guide](#troubleshooting)
3. Open an issue on GitHub
4. Contact the development team

## 📈 Roadmap

- [ ] Add model versioning
- [ ] Implement A/B testing
- [ ] Add model performance monitoring
- [ ] Create automated model retraining
- [ ] Add support for more model formats
- [ ] Implement caching for predictions
- [ ] Add real-time model updates

---

**Made with ❤️ by Mostafa Mahmoud**
