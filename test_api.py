"""
Quick API Test Script
Tests /api/predict and /api/predict/live endpoints
"""
import requests
import json

BASE_URL = "http://localhost:5000"

# Sample request payload
sample_data = {
    "latitude": 6.9271,
    "longitude": 79.8612,
    "temperature": 30,
    "humidity": 78,
    "visibility": 5,
    "weather": "Heavy Rain",
    "trafficDensity": 85,
    "speedLimit": 50,
    "roadType": "Galle Road (A2)",
    "areaType": "urban",
    "district": "Colombo",
    "city": "Colombo"
}

def test_health():
    print("\n--- Testing /api/health ---")
    r = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))

def test_predict():
    print("\n--- Testing /api/predict (POST) ---")
    try:
        r = requests.post(f"{BASE_URL}/api/predict", json=sample_data)
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print(f"ERROR: {e}")

def test_predict_live():
    print("\n--- Testing /api/predict/live (POST) ---")
    live_data = {
        "latitude": 6.9271,
        "longitude": 79.8612,
        "city": "Colombo",
        "district": "Colombo",
        "trafficDensity": 65,
        "speedLimit": 60,
        "roadType": "Main Road",
        "areaType": "urban"
    }
    try:
        r = requests.post(f"{BASE_URL}/api/predict/live", json=live_data)
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_health()
    test_predict()
    test_predict_live()
