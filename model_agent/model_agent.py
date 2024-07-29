from flask import Flask, request, jsonify
import requests
import json
from datetime import datetime, timedelta
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Configuration
CONFIG = {
    "model_path": "models/anomaly_detection_model.pkl",
    "metrics_pull_interval": 10,
    "alert_url": "http://alertmanager:9093/api/v1/alerts",
    "feedback_url": "http://feedbackmanager:5000/feedback"
}

# Load the model
def load_model():
    return joblib.load(CONFIG["model_path"])

# Save the model
def save_model(model):
    joblib.dump(model, CONFIG["model_path"])

# Model Inference Engine
class ModelInferenceEngine:
    def __init__(self, model):
        self.model = model

    def detect_anomaly(self, app_metrics, node_metrics):
        data = [app_metrics['cpu_usage'], app_metrics['memory_usage']]
        prediction = self.model.predict([data])
        return prediction[0]

# Alert Generator
class AlertGenerator:
    def __init__(self, alert_url):
        self.alert_url = alert_url

    def generate_alert(self, metrics):
        alert = {
            "labels": {"alertname": "AnomalyDetected", "severity": "critical"},
            "annotations": {"description": f"Anomaly detected with metrics: {metrics}"},
            "startsAt": datetime.utcnow().isoformat() + 'Z'
        }
        response = requests.post(self.alert_url, json=[alert])
        return response.status_code

# Feedback Handler
class FeedbackHandler:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    def retrain_model(self, feedback_data):
        X = []
        y = []
        for feedback in feedback_data:
            metrics = feedback['metrics']
            actual_label = feedback['actual_label']
            X.append([metrics['cpu_usage'], metrics['memory_usage']])
            y.append(1 if actual_label in ['true_positive', 'false_negative'] else 0)

        if X and y:
            model = RandomForestClassifier()
            model.fit(X, y)
            self.model_loader.save_model(model)

@app.route('/')
def index():
    return jsonify({"status": "running"}), 200

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    timestamp = data.get('timestamp')
    if not timestamp:
        return jsonify({"status": "error", "message": "Timestamp is required"}), 400

    try:
        datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid timestamp format"}), 400

    actual_label = data['actual_label']
    details = data.get('details', '')

    feedback_data = {
        "timestamp": timestamp,
        "actual_label": actual_label,
        "details": details
    }

    feedback_storage.append(feedback_data)
    feedback_handler.retrain_model(feedback_storage)  # Retrain model with new feedback

    return jsonify({"status": "success"}), 200

@app.route('/metrics', methods=['POST'])
def receive_metrics():
    data = request.get_json()
    app_metrics = data['app_metrics']
    node_metrics = data['node_metrics']

    # Process metrics for anomaly detection
    anomaly = inference_engine.detect_anomaly(app_metrics, node_metrics)
    if anomaly == 1:
        alert_generator.generate_alert(app_metrics)
    
    return jsonify({"status": "metrics received"}), 200

if __name__ == "__main__":
    model_loader = ModelLoader(CONFIG)
    model = model_loader.load_model()
    inference_engine = ModelInferenceEngine(model)
    alert_generator = AlertGenerator(CONFIG["alert_url"])
    feedback_handler = FeedbackHandler(model_loader)
    app.run(host='0.0.0.0', port=5000)
