"""
Prediction utilities for traffic accident prediction using XGBoost and LSTM models
"""
import numpy as np
import pickle
import json
import logging
from pathlib import Path
import xgboost as xgb
from tensorflow import keras

logger = logging.getLogger(__name__)

class AccidentPredictor:
    """
    Traffic accident predictor using ensemble of XGBoost and LSTM models
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the predictor with trained models
        
        Args:
            models_dir (str): Directory containing the trained models
        """
        self.models_dir = Path(models_dir)
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models and scaler"""
        try:
            # Load XGBoost model
            xgb_path = self.models_dir / 'xgboost_accident_predictor.json'
            if xgb_path.exists():
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(str(xgb_path))
                logger.info("XGBoost model loaded successfully")
            else:
                logger.warning(f"XGBoost model not found at {xgb_path}")
            
            # Load LSTM model
            lstm_path = self.models_dir / 'lstm_accident_predictor.h5'
            if lstm_path.exists():
                self.lstm_model = keras.models.load_model(str(lstm_path))
                logger.info("LSTM model loaded successfully")
            else:
                logger.warning(f"LSTM model not found at {lstm_path}")
            
            # Load scaler
            scaler_path = self.models_dir / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning(f"Scaler not found at {scaler_path}")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def preprocess_input(self, data):
        """
        Preprocess input data for prediction
        
        Args:
            data (dict): Raw input data
            
        Returns:
            np.ndarray: Preprocessed features
        """
        # Extract features from input data
        # Adjust these features based on your training data columns
        features = [
            data.get('hour', 12),
            data.get('day_of_week', 1),
            data.get('weather_condition', 1),  # 1: Clear, 2: Rain, 3: Fog, etc.
            data.get('road_condition', 1),      # 1: Dry, 2: Wet, 3: Icy, etc.
            data.get('speed_limit', 50),
            data.get('traffic_volume', 100),
            data.get('num_lanes', 2),
            data.get('lighting_condition', 1),  # 1: Daylight, 2: Dark, etc.
        ]
        
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_array = self.scaler.transform(features_array)
        
        return features_array
    
    def predict_xgboost(self, features):
        """
        Make prediction using XGBoost model
        
        Args:
            features (np.ndarray): Preprocessed features
            
        Returns:
            float: Accident probability
        """
        if self.xgb_model is None:
            raise ValueError("XGBoost model not loaded")
        
        dmatrix = xgb.DMatrix(features)
        prediction = self.xgb_model.predict(dmatrix)
        return float(prediction[0])
    
    def predict_lstm(self, features):
        """
        Make prediction using LSTM model
        
        Args:
            features (np.ndarray): Preprocessed features
            
        Returns:
            float: Accident probability
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model not loaded")
        
        # Reshape for LSTM (add time dimension)
        features_reshaped = features.reshape(1, 1, features.shape[1])
        prediction = self.lstm_model.predict(features_reshaped, verbose=0)
        return float(prediction[0][1])  # Probability of accident class
    
    def predict(self, data, use_ensemble=True):
        """
        Make accident prediction
        
        Args:
            data (dict): Input data containing traffic conditions
            use_ensemble (bool): Whether to use ensemble of both models
            
        Returns:
            dict: Prediction results with probability and risk level
        """
        try:
            # Preprocess input
            features = self.preprocess_input(data)
            
            # Get predictions from available models
            predictions = []
            model_results = {}
            
            if self.xgb_model is not None:
                xgb_prob = self.predict_xgboost(features)
                predictions.append(xgb_prob)
                model_results['xgboost'] = xgb_prob
            
            if self.lstm_model is not None:
                lstm_prob = self.predict_lstm(features)
                predictions.append(lstm_prob)
                model_results['lstm'] = lstm_prob
            
            if not predictions:
                raise ValueError("No models available for prediction")
            
            # Calculate final probability
            if use_ensemble and len(predictions) > 1:
                # Average ensemble
                final_probability = float(np.mean(predictions))
            else:
                final_probability = predictions[0]
            
            # Determine risk level
            if final_probability < 0.3:
                risk_level = "Low"
            elif final_probability < 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return {
                'success': True,
                'probability': round(final_probability, 4),
                'risk_level': risk_level,
                'model_results': model_results,
                'input_data': data
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def batch_predict(self, data_list):
        """
        Make predictions for multiple data points
        
        Args:
            data_list (list): List of input data dictionaries
            
        Returns:
            list: List of prediction results
        """
        return [self.predict(data) for data in data_list]
