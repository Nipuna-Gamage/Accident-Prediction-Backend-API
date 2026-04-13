# AI Traffic Accident Prediction API 🚦

AI-Driven Traffic Accident Prediction System using ensemble machine learning (XGBoost + LSTM) for Sri Lankan traffic conditions.

**Author:** Nipuna Gamage  
**Project:** IIT/UoW Final Year Project

---

## 🎯 Features

- **Real-time Prediction**: Instant accident risk assessment for any location
- **Ensemble ML**: Combines XGBoost and LSTM models for better accuracy
- **Temporal Analysis**: Considers time, day, rush hours, and seasonal patterns
- **Risk Scoring**: Multi-factor risk calculation (weather, visibility, traffic)
- **Hotspot Mapping**: Identifies high-risk accident zones
- **Batch Processing**: Analyze multiple locations simultaneously
- **Sri Lankan Context**: Optimized for local roads, weather, and traffic patterns

---

## 📁 Project Structure

```
traffic-accident-api/
├── app.py                          # Flask API (Advanced version with 21 features)
├── predict.py                      # Prediction utilities (legacy)
├── test_api.py                     # Comprehensive API testing
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── .env.example                    # Environment template
│
├── models/                         # Trained ML models (place here)
│   ├── xgboost_accident_predictor.json
│   ├── lstm_accident_predictor.h5
│   └── scaler.pkl
│
├── data/                           # Dataset directory
│   └── sl_traffic_accidents.csv
│
├── logs/                           # Application logs
│   └── api.log
│
└── frontend/                       # Frontend integration
    └── apiService.js
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- Flask 3.0+
- Flask-CORS
- NumPy
- Pandas
- XGBoost
- TensorFlow 2.15+
- scikit-learn

### 2. Add Trained Models

Place your trained models in the `models/` directory:
- `xgboost_accident_predictor.json`
- `lstm_accident_predictor.h5`
- `scaler.pkl`

### 3. Run the API

```bash
python app.py
```

The API will start on `http://localhost:5000`

You'll see:
```
================================================================================
AI TRAFFIC ACCIDENT PREDICTION API
================================================================================

✓ XGBoost model loaded
✓ LSTM model loaded
✓ Scaler loaded
✅ All models loaded successfully!

✅ Server ready to start!
📍 Running on: http://localhost:5000
```

### 4. Test the API

In a new terminal:

```bash
python test_api.py
```

---

## 📡 API Endpoints

### 1. Home / Info
```http
GET /
```

Returns API information and available endpoints.

**Response:**
```json
{
  "name": "AI Traffic Accident Prediction API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": { ... }
}
```

---

### 2. Health Check
```http
GET /api/health
```

Check API and model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2024-01-15T18:30:00"
}
```

---

### 3. Single Location Prediction ⭐
```http
POST /api/predict
```

Predict accident risk for a specific location.

**Request Body:**
```json
{
  "latitude": 6.9271,
  "longitude": 79.8612,
  "temperature": 32,
  "humidity": 78,
  "visibility": 4,
  "weather": "Heavy Rain",
  "trafficDensity": 85,
  "speedLimit": 50,
  "roadType": "Galle Road (A2)",
  "areaType": "urban",
  "district": "Colombo",
  "city": "Kollupitiya",
  "timestamp": "2024-01-15T18:30:00"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "xgboost": {
      "risk_level": "High Risk",
      "confidence": 87.5,
      "probability": 73.2
    },
    "lstm": {
      "risk_level": "High Risk",
      "confidence": 89.3,
      "probability": 76.8
    },
    "ensemble": {
      "risk_level": "High Risk",
      "probability": 75.0,
      "recommendation": {
        "level": "high",
        "message": "HIGH RISK: Drive with extreme caution...",
        "color": "#ea580c",
        "action": "CAUTION"
      }
    }
  },
  "location": {
    "latitude": 6.9271,
    "longitude": 79.8612,
    "city": "Kollupitiya",
    "district": "Colombo"
  },
  "timestamp": "2024-01-15T18:30:15"
}
```

---

### 4. Batch Prediction
```http
POST /api/predict/batch
```

Predict risk for multiple locations.

**Request Body:**
```json
{
  "locations": [
    { location_data_1 },
    { location_data_2 },
    ...
  ]
}
```

**Response:**
```json
{
  "success": true,
  "count": 3,
  "predictions": [
    {
      "location": { "city": "Galle Fort", ... },
      "risk_probability": 75.8,
      "risk_level": "High Risk",
      "recommendation": { ... }
    },
    ...
  ],
  "timestamp": "2024-01-15T18:30:00"
}
```

---

### 5. Accident Hotspots
```http
GET /api/hotspots
GET /api/hotspots?district=Colombo
```

Get high-risk accident zones.

**Response:**
```json
{
  "success": true,
  "count": 3,
  "hotspots": [
    {
      "city": "Kollupitiya",
      "district": "Colombo",
      "latitude": 6.9147,
      "longitude": 79.8501,
      "risk_probability": 87.5,
      "alert_count": 12
    },
    ...
  ]
}
```

---

### 6. System Statistics
```http
GET /api/stats
```

Get API usage and prediction statistics.

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_predictions_today": 247,
    "high_risk_alerts": 68,
    "active_hotspots": 12,
    "model_accuracy": 87.3
  },
  "districts": {
    "Colombo": { "predictions": 112, "high_risk": 34 },
    ...
  }
}
```

---

## 📊 Input Parameters (21 Features)

| Parameter | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `latitude` | float | Latitude coordinate | 6.9271 |
| `longitude` | float | Longitude coordinate | 79.8612 |
| `temperature` | float | Temperature (°C) | 32 |
| `humidity` | float | Humidity (%) | 78 |
| `visibility` | float | Visibility (0-10 km) | 4 |
| `weather` | string | Weather condition | "Clear", "Heavy Rain", "Fog" |
| `trafficDensity` | float | Traffic density (0-100) | 85 |
| `speedLimit` | float | Speed limit (km/h) | 50 |
| `roadType` | string | Road type | "Galle Road (A2)", "Main Road" |
| `areaType` | string | Area type | "urban", "rural", "suburban" |
| `district` | string | District | "Colombo", "Galle", "Matara" |
| `city` | string | City name | "Kollupitiya" |  
| `timestamp` | string | ISO datetime (optional) | "2024-01-15T18:30:00" |

### Weather Values
- `Clear`, `Partly Cloudy`, `Cloudy`, `Light Rain`, `Heavy Rain`, `Fog`

### Road Types
- `Rural Road`, `Urban Street`, `Main Road`, `Galle Road (A2)`, `Southern Expressway`

### Area Types
- `rural`, `suburban`, `urban`, `tourist`

### Districts
- `Colombo`, `Kalutara`, `Galle`, `Matara`

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

Tests include:
- ✅ Home endpoint
- ✅ Health check
- ✅ Single prediction
- ✅ Batch prediction
- ✅ Hotspots API
- ✅ Stats API
- ✅ Error handling

---

## 🎨 Frontend Integration

Use the provided `frontend/apiService.js`:

```javascript
import apiService from './frontend/apiService';

// Make prediction
const result = await apiService.predictAccident({
  latitude: 6.9271,
  longitude: 79.8612,
  temperature: 32,
  weather: "Heavy Rain",
  // ... other parameters
});

console.log(`Risk: ${result.prediction.ensemble.probability}%`);
console.log(result.prediction.ensemble.recommendation.message);
```

---

## 🔧 Configuration

Create a `.env` file (see `.env.example`):

```env
PORT=5000
DEBUG=False
MODELS_DIR=models
DATA_DIR=data
LOGS_DIR=logs
```

---

## 🏗️ Model Architecture

The system uses an **ensemble approach**:

1. **XGBoost Classifier**: Gradient boosting for structured features
2. **LSTM Neural Network**: Sequential pattern recognition
3. **Feature Engineering**: 21 engineered features including:
   - Temporal features (hour, day, month, weekend, rush hour)
   - Weather severity encoding
   - Road risk scoring
   - Traffic and visibility risk scores

**Final Prediction**: Average of both model probabilities

---

## 📈 Risk Levels

| Probability | Risk Level | Recommendation |
|-------------|-----------|----------------|
| 80-100% | Extreme | AVOID route |
| 60-79% | High | CAUTION - reduce speed |
| 40-59% | Moderate | ALERT - stay vigilant |
| 0-39% | Low | PROCEED normally |

---

## 🐛 Troubleshooting

**Models not loading?**
```bash
# Check if model files exist
ls -la models/

# Verify file paths in app.py
```

**Port already in use?**
```bash
# Change port in app.py or .env
PORT=5001
```

**Import errors?**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## 📝 License

This project is part of an IIT/UoW Final Year Project.

---

## ✨ Credits

**Developer**: Nipuna Gamage  
**Institution**: IIT/University of Westminster  
**Year**: 2024

---

## 🔮 Future Enhancements

- [ ] Database integration for historical data
- [ ] Real-time weather API integration
- [ ] Google Maps integration
- [ ] Mobile app (Flutter)
- [ ] User feedback system
- [ ] Model retraining pipeline

---

**Happy Predicting! Stay Safe! 🚗💨**
