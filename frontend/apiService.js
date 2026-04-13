/**
 * Frontend API Integration
 * Connect React Web App to Flask Backend
 */

// ============================================================================
// API Configuration
// ============================================================================

const API_CONFIG = {
    baseURL: 'http://localhost:5000',
    endpoints: {
        health: '/api/health',
        predict: '/api/predict',
        batchPredict: '/api/predict/batch',
        hotspots: '/api/hotspots',
        stats: '/api/stats'
    }
};


// ============================================================================
// API Service Class
// ============================================================================

class TrafficAPIService {
    constructor(baseURL = API_CONFIG.baseURL) {
        this.baseURL = baseURL;
    }

    /**
     * Generic API call method
     */
    async apiCall(endpoint, options = {}) {
        try {
            const response = await fetch(`${this.baseURL}${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API Call Failed:', error);
            throw error;
        }
    }

    /**
     * Check API health
     */
    async checkHealth() {
        return await this.apiCall(API_CONFIG.endpoints.health);
    }

    /**
     * Predict accident risk for single location
     */
    async predictRisk(locationData) {
        return await this.apiCall(API_CONFIG.endpoints.predict, {
            method: 'POST',
            body: JSON.stringify(locationData)
        });
    }

    /**
     * Batch prediction for multiple locations
     */
    async batchPredict(locations) {
        return await this.apiCall(API_CONFIG.endpoints.batchPredict, {
            method: 'POST',
            body: JSON.stringify({ locations })
        });
    }

    /**
     * Get accident hotspots
     */
    async getHotspots(district = null) {
        const url = district
            ? `${API_CONFIG.endpoints.hotspots}?district=${district}`
            : API_CONFIG.endpoints.hotspots;
        return await this.apiCall(url);
    }

    /**
     * Get system statistics
     */
    async getStats() {
        return await this.apiCall(API_CONFIG.endpoints.stats);
    }
}


// ============================================================================
// React Hook for API Integration
// ============================================================================

/**
 * Custom React Hook for Traffic API
 * Usage in React components
 */

// Example: usePrediction.js
/*
import { useState, useEffect } from 'react';
import { TrafficAPIService } from './apiService';

export const usePrediction = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  
  const api = new TrafficAPIService();

  const predictRisk = async (locationData) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await api.predictRisk(locationData);
      setPrediction(result);
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  };

  return { predictRisk, prediction, loading, error };
};
*/


// ============================================================================
// Example React Component Integration
// ============================================================================

/**
 * Example: Real-time Prediction Component
 */

const ExamplePredictionComponent = `
import React, { useState } from 'react';
import { TrafficAPIService } from './apiService';

const RiskPredictionForm = () => {
  const [formData, setFormData] = useState({
    latitude: 6.9271,
    longitude: 79.8612,
    city: 'Colombo',
    district: 'Colombo',
    temperature: 30,
    humidity: 75,
    visibility: 8,
    weather: 'Clear',
    trafficDensity: 50,
    speedLimit: 50,
    roadType: 'Main Road',
    areaType: 'urban'
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const api = new TrafficAPIService();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const result = await api.predictRisk(formData);
      setPrediction(result);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="prediction-form">
      <form onSubmit={handleSubmit}>
        {/* Form inputs here */}
        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict Risk'}
        </button>
      </form>

      {prediction && (
        <div className="prediction-result">
          <h3>Prediction Result</h3>
          <div className="ensemble-result">
            <p>Risk Level: {prediction.prediction.ensemble.risk_level}</p>
            <p>Probability: {prediction.prediction.ensemble.probability}%</p>
            <p>Recommendation: {prediction.prediction.ensemble.recommendation.message}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default RiskPredictionForm;
`;


// ============================================================================
// Real-time Dashboard Integration
// ============================================================================

/**
 * Dashboard Component with Auto-refresh
 */

const ExampleDashboardComponent = `
import React, { useState, useEffect } from 'react';
import { TrafficAPIService } from './apiService';

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [hotspots, setHotspots] = useState([]);
  const api = new TrafficAPIService();

  // Fetch data on component mount
  useEffect(() => {
    fetchDashboardData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [statsData, hotspotsData] = await Promise.all([
        api.getStats(),
        api.getHotspots()
      ]);
      
      setStats(statsData.stats);
      setHotspots(hotspotsData.hotspots);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    }
  };

  return (
    <div className="dashboard">
      {/* Stats Cards */}
      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <h3>Total Predictions</h3>
            <p>{stats.total_predictions_today}</p>
          </div>
          <div className="stat-card">
            <h3>High Risk Alerts</h3>
            <p>{stats.high_risk_alerts}</p>
          </div>
          <div className="stat-card">
            <h3>Active Hotspots</h3>
            <p>{stats.active_hotspots}</p>
          </div>
          <div className="stat-card">
            <h3>Model Accuracy</h3>
            <p>{stats.model_accuracy}%</p>
          </div>
        </div>
      )}

      {/* Hotspots List */}
      <div className="hotspots-section">
        <h2>Accident Hotspots</h2>
        {hotspots.map((hotspot, index) => (
          <div key={index} className="hotspot-card">
            <h4>{hotspot.city}, {hotspot.district}</h4>
            <p>Risk: {hotspot.risk_probability}%</p>
            <p>Alerts: {hotspot.alert_count}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Dashboard;
`;


// ============================================================================
// Map Integration with Predictions
// ============================================================================

/**
 * Google Maps integration with risk predictions
 */

const MapIntegrationExample = `
import React, { useState, useEffect } from 'react';
import { GoogleMap, Marker, Circle } from '@react-google-maps/api';
import { TrafficAPIService } from './apiService';

const RiskHeatmap = () => {
  const [hotspots, setHotspots] = useState([]);
  const api = new TrafficAPIService();

  useEffect(() => {
    loadHotspots();
  }, []);

  const loadHotspots = async () => {
    const data = await api.getHotspots();
    setHotspots(data.hotspots);
  };

  const getMarkerColor = (risk) => {
    if (risk >= 70) return '#dc2626'; // Red
    if (risk >= 50) return '#f59e0b'; // Yellow
    return '#16a34a'; // Green
  };

  return (
    <GoogleMap
      center={{ lat: 6.9271, lng: 79.8612 }}
      zoom={10}
    >
      {hotspots.map((spot, index) => (
        <React.Fragment key={index}>
          <Marker
            position={{ lat: spot.latitude, lng: spot.longitude }}
            icon={{
              path: window.google.maps.SymbolPath.CIRCLE,
              fillColor: getMarkerColor(spot.risk_probability),
              fillOpacity: 0.8,
              strokeWeight: 2,
              strokeColor: '#ffffff',
              scale: 10
            }}
          />
          
          <Circle
            center={{ lat: spot.latitude, lng: spot.longitude }}
            radius={500}
            options={{
              fillColor: getMarkerColor(spot.risk_probability),
              fillOpacity: 0.2,
              strokeWeight: 1
            }}
          />
        </React.Fragment>
      ))}
    </GoogleMap>
  );
};

export default RiskHeatmap;
`;


// ============================================================================
// WebSocket for Real-time Updates (Future Enhancement)
// ============================================================================

/**
 * WebSocket connection for live alerts
 */

const WebSocketExample = `
// Backend (Flask-SocketIO)
from flask_socketio import SocketIO, emit

socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_response', {'status': 'connected'})

@socketio.on('request_prediction')
def handle_prediction(data):
    # Process prediction
    result = process_prediction(data)
    emit('prediction_result', result)


// Frontend (React)
import io from 'socket.io-client';

const socket = io('http://localhost:5000');

socket.on('connect', () => {
  console.log('Connected to WebSocket');
});

socket.on('prediction_result', (data) => {
  console.log('Real-time prediction:', data);
  // Update UI with prediction
});

socket.emit('request_prediction', locationData);
`;


// ============================================================================
// Error Handling & Retry Logic
// ============================================================================

class APIServiceWithRetry extends TrafficAPIService {
    async apiCallWithRetry(endpoint, options = {}, maxRetries = 3) {
        let lastError;

        for (let i = 0; i < maxRetries; i++) {
            try {
                return await this.apiCall(endpoint, options);
            } catch (error) {
                lastError = error;
                console.log(`Retry ${i + 1}/${maxRetries}...`);
                await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
            }
        }

        throw lastError;
    }
}


// ============================================================================
// Export
// ============================================================================

// For use in React app
export {
    TrafficAPIService,
    APIServiceWithRetry,
    API_CONFIG
};

// Example exports for documentation
export const examples = {
    PredictionComponent: ExamplePredictionComponent,
    DashboardComponent: ExampleDashboardComponent,
    MapIntegration: MapIntegrationExample,
    WebSocketExample
};

console.log('API Integration module loaded successfully!');