from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import requests
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages

# URL of the model agent
MODEL_AGENT_URL = "http://model-agent-service:5000/submit_feedback"

@app.route('/')
def index():
    return render_template('feedback_form.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    timestamp = request.form.get('timestamp')
    actual_label = request.form.get('actual_label')
    details = request.form.get('details')

    feedback_data = {
        "timestamp": timestamp,
        "actual_label": actual_label,
        "details": details
    }

    response = requests.post(MODEL_AGENT_URL, json=feedback_data)

    if response.status_code == 200:
        flash('Feedback submitted successfully!', 'success')
    else:
        flash(f'Failed to submit feedback: {response.status_code}', 'danger')

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
