# ⚙️ Industrial Predictive Maintenance – End-to-End MLOps Pipeline

An end-to-end Machine Learning deployment project for predicting industrial machine failure from sensor readings.

The project transforms a trained Random Forest model into a production-style ML application using **FastAPI, Pydantic, Docker, Streamlit, Pytest, and GitHub Actions**.

---

## 🎯 Project Objective

The goal is to move a machine learning model from a notebook environment into a reliable and reproducible prediction service.

The system accepts machine sensor readings, validates the incoming data, performs the same feature engineering used during training, and returns a machine-failure prediction with its probability.

---

## 🏗️ Architecture

```text
Machine Sensor Data
        ↓
Streamlit Frontend
        ↓
FastAPI REST API
        ↓
Pydantic Input Validation
        ↓
Feature Engineering
        ↓
Random Forest Model
        ↓
Decision Threshold (0.70)
        ↓
Failure / No Failure Prediction
```

The complete application can also be packaged and executed inside a **Docker container**.

---

## 🤖 Machine Learning Model

The prediction model was developed using the **AI4I 2020 Predictive Maintenance Dataset**.

Model:

- Random Forest Classifier
- Binary classification:
  - `0` → No Failure
  - `1` → Machine Failure

The model uses a decision threshold of:

```text
0.70
```

This operating threshold was selected on validation data to reduce false maintenance alarms while retaining useful failure detection.

### Held-Out Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 99.00% |
| Precision | 100.00% |
| Recall | 70.59% |
| F1 Score | 82.76% |
| False Discovery Rate | 0.00% |

Confusion matrix:

```text
True Positives  = 36
False Positives = 0
False Negatives = 15
True Negatives  = 1449
```

> Zero false positives were observed on the held-out test set; this does not imply that future production data will always have zero false alarms.

---

## 📥 API Input

The API receives raw machine measurements:

```json
{
  "type": "L",
  "air_temperature": 300.0,
  "process_temperature": 310.0,
  "rotational_speed": 1500,
  "torque": 40.0,
  "tool_wear": 100
}
```

Machine type accepts:

```text
L
M
H
```

---

## 🧮 Feature Engineering

Before prediction, the API recreates the engineered features used during model training.

Examples include:

```text
Temperature Difference
Power
Torque × Tool Wear Interaction
Machine Type One-Hot Encoding
```

Power is calculated from torque and rotational speed:

```text
Power = Torque × RPM × 2π / 60
```

The final model receives the features in the same order used during training to avoid training-serving skew.

---

## 🔌 FastAPI Model Serving

The trained model is exposed through a REST API built with **FastAPI**.

Main endpoints:

```text
GET  /
GET  /health
POST /predict
```

Run the API locally:

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Swagger UI can be used to interactively test the prediction endpoint.

---

## 🛡️ Input Schema Validation

Incoming requests are validated using **Pydantic**.

Invalid inputs are rejected before reaching the ML model.

Examples include:

- Text in numerical sensor fields
- Unsupported machine types
- Missing required values
- Invalid request structures

Invalid requests return:

```text
400 Bad Request
```

This prevents malformed data from reaching the prediction pipeline.

---

## 🖥️ Streamlit Frontend

A simple Streamlit interface allows users to enter machine sensor readings and receive predictions without manually sending API requests.

Run it with:

```bash
streamlit run app.py
```

The interface displays:

- Machine failure prediction
- Failure probability
- Decision threshold

---

## 🐳 Docker Containerization

The API is containerized using Docker so that the application can run consistently across environments.

Build the Docker image:

```bash
docker build -t predictive-maintenance-api .
```

Run the container:

```bash
docker run --name predictive-maintenance-container -p 8000:8000 predictive-maintenance-api
```

Then access:

```text
http://127.0.0.1:8000/docs
```

---

## 🧪 Automated Testing

API tests are implemented using **Pytest**.

The test suite checks:

- Health endpoint
- Valid prediction request
- Invalid numerical sensor input
- Invalid machine type

Run the tests:

```bash
pytest -v
```

Current local test result:

```text
4 passed
```

---

## 🔄 CI/CD with GitHub Actions

A GitHub Actions workflow is included in:

```text
.github/workflows/tests.yml
```

On configured repository events, the workflow automatically installs the required dependencies and executes the test suite.

This helps detect API or validation regressions before changes are accepted.

---

## 📁 Project Structure

```text
predictive-maintenance-mlops/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── main.py
├── app.py
├── test_main.py
├── predictive_maintenance_model.pkl
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

### File Description

`main.py` — FastAPI application, validation, preprocessing and model inference.

`app.py` — Streamlit user interface.

`predictive_maintenance_model.pkl` — Serialized trained Random Forest model and deployment metadata.

`test_main.py` — Automated API tests.

`Dockerfile` — Container configuration.

`requirements.txt` — Python dependencies.

`tests.yml` — GitHub Actions CI workflow.

---

## 🛠️ Technology Stack

- Python
- Scikit-learn
- FastAPI
- Pydantic
- Pandas
- NumPy
- Streamlit
- Docker
- Pytest
- Git
- GitHub Actions

---

## 🚀 Skills Demonstrated

This project demonstrates:

- Machine Learning Model Serving
- REST API Development
- Input Schema Validation
- Training-Serving Feature Consistency
- Model Serialization
- Docker Containerization
- Automated API Testing
- Continuous Integration
- Interactive ML Application Development
- End-to-End MLOps Workflow

---

## 👩‍💻 Author

**Shrouk Ahmed**

Machine Learning / Data Science Portfolio Project