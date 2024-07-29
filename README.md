## Enhancing Kubernetes Security with Real-Time Anomaly Detection and Feedback-Driven Machine Learning Models
#### Overview
This repository contains the code and resources for our project on enhancing Kubernetes security through real-time anomaly detection and feedback-driven machine learning models. The project includes two main applications:

- Feedback Application: A web-based interface for users to submit feedback on detected anomalies.
- Model Agent: An application that collects real-time metrics from Kubernetes nodes and applications, performs anomaly detection using pre-trained ML models, generates alerts, and retrains the models based on user feedback.

------------


#### Feedback Application
The Feedback Application is a critical component of our framework, providing a user-friendly interface for submitting feedback on detected anomalies. This application enables users to provide valuable input that is used to retrain and improve the machine learning models employed by the Model Agent.

##### Key Features
- User-Friendly Interface: Simplifies the feedback submission process.
- Timestamp Input: Allows users to specify the time of the anomaly.
- Feedback Details: Collects detailed information to improve model accuracy.

##### Setup and Installation
1. Navigate to the f- eedback_app directory:
`cd feedback_app`

2. Install the required Python packages:
`pip install -r requirements.txt`

3. Run the application:
`python app.py`

##### Docker Setup
1. Build the Docker image:
`docker build -f ../Dockerfile_feedback -t feedback_app .`

2. Run the Docker container:
`docker run -p 5000:5000 feedback_app`

------------

#### Model Agent
The Model Agent is responsible for real-time anomaly detection and model retraining. It performs several critical functions, including metrics collection, anomaly detection, alert generation, and feedback handling.

##### Key Features
Real-Time Detection: Analyzes metrics in real-time to identify potential attacks.
Adaptive Learning: Continuously improves model performance based on user feedback.
Integration with Kubernetes: Seamlessly integrates with Kubernetes for metrics collection and alerting.

##### Setup and Installation
1. Navigate to the model_agent directory:
`cd model_agent
`
2. Install the required Python packages:
`pip install -r requirements.txt
`
3. Run the Model Agent:
`python agent.py
`
##### Docker Setup
1. Build the Docker image:
`docker build -f ../Dockerfile_model -t model_agent .`

2. Run the Docker container:
`docker run -p 5001:5001 model_agent`

##### Kubernetes Deployment
1. Navigate to the kubernetes directory:
`cd kubernetes`

2. Apply the Feedback Application deployment:
`kubectl apply -f feedback_deployment.yaml
`
3. Apply the Model Agent deployment:
`kubectl apply -f model_agent_deployment.yaml
`
------------


#### Usage
- Feedback Application: Access the web interface through the specified port (default: 5000) to submit feedback on detected anomalies.
- Model Agent: Collects metrics, detects anomalies, generates alerts, and retrains models based on the feedback received.