# PlantCare ML Cycle 

An end-to-end machine learning application for detecting plant diseases from leaf images. The project combines image classification, model evaluation, a FastAPI prediction service, monitoring, user feedback and retraining management.

## 1. Project Overview

PlantCare ML Cycle uses a deep learning model to classify plant leaf images into 27 disease and healthy-plant categories across 9 crops.

The system allows users to:
- Upload a plant leaf image and receive a disease prediction.
- View prediction confidence and the top three predicted classes.
- Submit feedback on prediction correctness.
- Monitor API health and prediction statistics.
- Visualise prediction patterns and confidence distributions.
- Monitor model performance and check whether retraining is recommended.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Machine learning | TensorFlow / Keras |
| Model architecture | MobileNetV2 |
| Image processing | Pillow, NumPy |
| API | FastAPI |
| Dashboard | Streamlit |
| Visualisation | Plotly |
| Data processing | Pandas |
| Development environment | Google Colab |

## 3. Model and Dataset

**Dataset:** [Plant Disease Detection Dataset – Kaggle](https://www.kaggle.com/datasets/mgmitesh/plant-disease-detection-dataset)

The dataset contains images organised into training, validation and test splits:

- Training: 49,984 images
- Validation: 12,494 images
- Test: 39 images
- Number of crops: 9
- Number of classes: 27

The dataset is distributed under the Creative Commons Attribution 4.0 (CC BY 4.0) licence. Refer to the original dataset page for attribution and usage details.

### Model Architecture

The selected model uses MobileNetV2 pretrained on ImageNet as its feature extraction backbone.

The architecture includes:
- MobileNetV2 feature extraction.
- Image augmentation.
- Global Average Pooling.
- Dropout regularisation.
- A Dense softmax classification layer with 27 outputs.

Input images are resized to 256 × 256 pixels and processed as RGB images.

### Validation Results

| Metric | Result |
|---|---:|
| Validation accuracy | 96.12% |
| Validation loss | 0.1184 |
| Macro F1-score | 0.9614 |
| Weighted F1-score | 0.9611 |
| Model parameters | 2,292,571 |
| Model size | Approximately 9.58 MB |

The selected MobileNetV2 model improved validation accuracy by 36.93 percentage points compared with the baseline CNN, which achieved 59.19%.

**Evaluation note:** The test split contains only 39 images. The reported performance metrics above are validation results, not test-set results.

## 4. System Architecture

The project follows this workflow:

1. **Data acquisition:** Obtain the plant disease image dataset.
2. **Preprocessing:** Organise image manifests, resize images and prepare training and validation datasets.
3. **Model development:** Train and evaluate the baseline CNN and MobileNetV2 models.
4. **Model selection:** Save the best-performing MobileNetV2 checkpoint.
5. **Prediction API:** Serve image predictions through FastAPI.
6. **Monitoring:** Record predictions, confidence scores and feedback.
7. **Retraining management:** Evaluate labelled prediction feedback against predefined performance criteria.
8. **Dashboard:** Display system status, predictions, analytics and retraining eligibility.

## 5. API Endpoints

The FastAPI application provides the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API and model status |
| POST | `/predict` | Predict plant disease from an uploaded image |
| POST | `/feedback` | Submit actual-class feedback |
| GET | `/metrics` | Retrieve prediction and monitoring metrics |
| GET | `/retrain-check` | Evaluate retraining eligibility |
| POST | `/retrain` | Queue an eligible retraining request |

### Retraining Management

The retraining eligibility policy uses two conditions:

- At least 20 labelled predictions must be collected.
- Monitored accuracy must fall below 90% before retraining is recommended.

When insufficient feedback is available, the system prevents a retraining request.

The API records eligible retraining requests in a monitoring log. Actual model training and deployment are performed separately through the development workflow; queuing a request does not automatically replace the deployed model.

## 6. Dashboard

The Streamlit dashboard provides:

- API and model health monitoring.
- Image upload and disease prediction.
- Top-three prediction results.
- Prediction feedback submission.
- Prediction distribution visualisation.
- Confidence distribution visualisation.
- Recent prediction history.
- Retraining eligibility and request management.

## 7. Project Structure

```text
PlantCare_ML_Cycle/
│
├── api/
│   └── api.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── processed/
│       └── class_mapping.json
│
├── models/
│   └── checkpoints/
│       └── best_plantcare_mobilenetv2.keras
│
├── notebooks/
│   └── PlantCare_ML_Cycle.ipynb
│
├── reports/
│   ├── mobilenetv2/
│   │   ├── classification_report.csv
│   │   ├── confusion_matrix.csv
│   │   ├── evaluation_metrics.json
│   │   ├── misclassification_analysis.csv
│   │   └── model_metadata.json
│   │
│   └── monitoring/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 8. Installation and Execution

### Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd PlantCare_ML_Cycle
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure project files

Obtain the dataset from Kaggle and follow the data preparation instructions in the notebook.

Ensure that the trained model checkpoint and class mapping file are available in their expected project directories.

### Start the FastAPI service

From the project root:

```bash
cd api
uvicorn api:app --host 127.0.0.1 --port 8000
```

### Start the Streamlit dashboard

In a separate terminal, from the project root:

```bash
streamlit run dashboard/app.py
```

The dashboard communicates with the API at `http://127.0.0.1:8000`.

## 9. Limitations and Future Improvements

- The test split is relatively small and should be expanded for stronger independent evaluation.
- Monitored accuracy depends on users submitting feedback.
- The retraining workflow currently queues requests rather than automatically training and deploying a new model.
- Future improvements could include automated retraining pipelines, model versioning, deployment automation and expanded crop coverage.

## 10. Author

Developed as an end-to-end machine learning lifecycle project.

**Project:** PlantCare ML Cycle  
**Primary model:** MobileNetV2  
**Application:** FastAPI and Streamlit

