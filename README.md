# PlantCare ML Cycle
### An End-to-End Plant Disease Detection and Machine Learning Monitoring System

## 1. Project Overview

PlantCare ML Cycle is an image-based machine learning application designed to identify plant diseases from leaf images.

The project implements an end-to-end machine learning workflow, covering dataset acquisition, image preprocessing, model development, evaluation, prediction through a Python API, monitoring through an interactive dashboard, and model retraining.

The system uses a MobileNetV2-based image classification model to classify plant leaves into 27 disease and healthy-plant categories across nine crops.

## 2. Project Objectives

The main objectives are to:

- Acquire and preprocess a plant disease image dataset.
- Develop and evaluate an image classification model.
- Provide predictions through a Python REST API.
- Monitor model health and prediction activity.
- Collect user feedback on predictions.
- Implement a retraining workflow to support model updates.
- Containerize the application using Docker.
- Evaluate API performance through load testing.

## 3. Technologies Used

| Component | Technology |
|---|---|
| Programming language | Python |
| Development environment | Google Colab |
| Deep learning framework | TensorFlow / Keras |
| Model architecture | MobileNetV2 |
| Image processing | PIL, TensorFlow |
| API | FastAPI |
| Dashboard | Streamlit |
| API communication | HTTP / REST |
| Containerization | Docker |
| Container orchestration | Docker Compose |
| Load testing | Locust |
| Version control | Git and GitHub |
| Dataset | Kaggle |

## 4. Dataset

The project uses the Plant Disease Detection Dataset from Kaggle:

**Dataset:** `mgmitesh/plant-disease-detection-dataset`

**License:** CC BY 4.0

The dataset contains images from nine crop categories and 27 classification classes.

### Dataset distribution

| Dataset split | Number of images |
|---|---:|
| Training | 49,984 |
| Validation | 12,494 |
| Testing | 39 |
| **Total** | **62,517** |

The dataset includes the following crops:

- Apple
- Cherry
- Corn
- Grape
- Peach
- Pepper
- Potato
- Strawberry
- Tomato

The raw dataset is not included in this GitHub repository because of its large size. Users should obtain it from Kaggle and place it in the appropriate project data directory.

## 5. Model Development and Evaluation

The project explored a baseline CNN model and a MobileNetV2-based model.

MobileNetV2 was selected for the final disease classification system.

### Model specifications

- Architecture: MobileNetV2
- Input image dimensions: 256 × 256 pixels
- Input channels: RGB
- Number of output classes: 27
- Model file: `best_plantcare_mobilenetv2.keras`

### Evaluation results

The final MobileNetV2 model achieved the following validation results:

| Metric | Result |
|---|---:|
| Validation accuracy | 96.12% |
| Validation loss | 0.1184 |
| Macro F1-score | 0.9614 |
| Weighted F1-score | 0.9611 |

The baseline CNN achieved approximately 59.19% validation accuracy.

These results demonstrate the improvement obtained using the MobileNetV2 architecture.

## 6. System Architecture

The application consists of three main components:

1. **Machine Learning Model:** Processes uploaded plant leaf images and predicts their classification.
2. **FastAPI Backend:** Provides prediction, feedback, monitoring, and retraining endpoints.
3. **Streamlit Dashboard:** Provides a user interface for interacting with the application and viewing monitoring information.

Docker and Docker Compose are used to run the application components in containers. HAProxy provides a load-balancing entry point for the API.

### Application workflow

1. A user uploads a plant leaf image through the dashboard.
2. The dashboard sends the image to the FastAPI prediction endpoint.
3. The API preprocesses the image and passes it to the trained model.
4. The model predicts the plant disease class and confidence score.
5. The API returns the prediction result.
6. Prediction information is recorded for monitoring and analysis.
7. Users can submit feedback on prediction correctness.

## 7. Application Features

### Image Prediction

Users can upload an individual plant leaf image and receive:

- Predicted disease or healthy classification.
- Prediction confidence score.
- Prediction ID.
- Top prediction results.

### System Monitoring

The dashboard provides monitoring information, including API health, model availability, prediction activity, and performance metrics.

### Data Visualization

The dashboard includes visualizations such as:

- Distribution of predicted plant diseases.
- Prediction confidence distribution.
- Recent prediction activity.

### User Feedback

Users can submit feedback indicating whether a prediction was correct. Feedback is recorded for monitoring and potential future model improvement.

### Model Retraining

The project includes a retraining workflow designed to support model updates using newly uploaded training data.

The workflow includes:

- Training data upload endpoint.
- Retraining eligibility checks.
- Candidate model training.
- Candidate model evaluation.
- Model acceptance checks.
- Model backup and replacement.

A retraining experiment was performed using a sample of 540 images, with up to 20 images per class.

The experiment achieved:

| Metric | Result |
|---|---:|
| Baseline accuracy | 99.07% |
| Candidate accuracy | 99.07% |
| Training epochs | 2 |

The candidate model was accepted under the configured acceptance criteria.

The retraining experiment demonstrates the workflow, but the sample was drawn from the original training dataset and should not be interpreted as independent real-world performance validation.

**Note:** The bulk-upload workflow is implemented, but its end-to-end operation still requires further verification.

## 8. API Endpoints

The FastAPI backend exposes the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks API and model health |
| `/predict` | POST | Accepts an image and returns a prediction |
| `/feedback` | POST | Records prediction feedback |
| `/metrics` | GET | Returns monitoring metrics |
| `/retrain-check` | GET | Checks retraining eligibility |
| `/retrain` | POST | Initiates model retraining |
| `/retraining/upload` | POST | Accepts retraining data |
| `/retraining/run` | POST | Runs the retraining workflow |

Interactive API documentation is available through FastAPI's Swagger UI at:

`http://localhost:8000/docs`

## 9. Docker Deployment

The project includes Docker configuration files:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `haproxy.cfg`

The application was built and tested locally using Docker Desktop and Docker Compose.

### Running the application

Clone the repository:

```bash
git clone https://github.com/kylaochweri/PlantCare-ML-Cycle.git
```

Navigate into the project:

```bash
cd PlantCare-ML-Cycle
```

Ensure that the trained model is available at:

```text
models/checkpoints/best_plantcare_mobilenetv2.keras
```

Ensure the required application dependencies and model files are available, then build and start the containers:

```bash
docker compose up --build -d
```

Check the running containers:

```bash
docker compose ps
```

### Accessing the application

- Streamlit Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

To stop the application:

```bash
docker compose down
```

**Deployment status:** Local Docker deployment was successfully tested. Public cloud deployment was not completed.

## 10. Load Testing with Locust

Locust was used to evaluate the API's performance under simulated concurrent user activity.

### Test configuration

| Parameter | Value |
|---|---:|
| Simulated users | 5 |
| Spawn rate | 1 user per second |
| Test duration | 30 seconds |
| Total requests | 84 |

### Load testing results

| Metric | Result |
|---|---:|
| Total requests | 84 |
| Failed requests | 0 |
| Average response time | 164.5 ms |
| Median response time | 130 ms |
| Minimum response time | 90.9 ms |
| Maximum response time | 1,090.3 ms |
| 95th percentile | 310 ms |
| Throughput | 2.945 requests/second |

The test recorded zero failed requests during the measured run.

**Limitation:** The performance evaluation covered one load configuration. A comparison involving one, two, and three API containers was not completed.

## 11. Project Structure

The repository is organized into the following main directories and files:

```text
PlantCare-ML-Cycle/
│
├── api/
│   └── api.py
│
├── dashboard/
│   └── app.py
│
├── notebook/
│   └── PlantCare_ML_Cycle.ipynb
│
├── models/
│   └── checkpoints/
│       └── best_plantcare_mobilenetv2.keras
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── incoming_training/
│
├── reports/
│   └── monitoring/
│
├── Dockerfile
├── docker-compose.yml
├── haproxy.cfg
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

Some generated files and local dataset files are excluded from version control.

## 12. Future Improvements

Potential future improvements include:

- Deploying the application to a public cloud platform.
- Completing load testing across multiple API container configurations.
- Improving bulk training-data upload and validation.
- Integrating model explainability visualizations into the dashboard.
- Evaluating the model using independent field images.
- Expanding the dataset to include more crops and disease categories.
- Improving automated model monitoring and retraining decisions.

## 13. Conclusion

PlantCare ML Cycle demonstrates an end-to-end machine learning application for plant disease classification.

The project combines image preprocessing, deep learning, model evaluation, REST API development, interactive monitoring, user feedback, retraining workflows, Docker containerization, and performance testing.

The final system demonstrates how a trained image classification model can be integrated into an application with monitoring and model lifecycle management capabilities.

## 14. Author

**Kyla Nyaboke Ochweri**

GitHub: [@kylaochweri](https://github.com/kylaochweri)

Project Repository: [PlantCare ML Cycle](https://github.com/kylaochweri/PlantCare-ML-Cycle)
Youtube unlisted demo link; https://youtu.be/UAy1C9kHaf8

