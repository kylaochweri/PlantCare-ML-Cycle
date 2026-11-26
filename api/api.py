
import csv
import io
import json
import os
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, UploadFile


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "checkpoints",
    "best_plantcare_mobilenetv2.keras"
)

MAPPING_PATH = os.path.join(
    PROJECT_DIR, "data", "processed", "class_mapping.json"
)

MONITORING_DIR = os.path.join(
    PROJECT_DIR, "reports", "monitoring"
)

PREDICTION_LOG = os.path.join(
    MONITORING_DIR, "predictions.csv"
)

FEEDBACK_LOG = os.path.join(
    MONITORING_DIR, "feedback.csv"
)

IMAGE_SIZE = (256, 256)
MODEL_NAME = "plantcare_mobilenetv2"
START_TIME = time.time()

os.makedirs(MONITORING_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD MODEL AND CLASS MAPPING
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(MAPPING_PATH):
    raise FileNotFoundError(f"Mapping not found: {MAPPING_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(MAPPING_PATH, "r") as file:
    class_to_index = json.load(file)

index_to_class = {
    int(index): class_name
    for class_name, index in class_to_index.items()
}

if len(index_to_class) != 27 or model.output_shape[-1] != 27:
    raise ValueError("Model and class mapping do not match.")


# --------------------------------------------------
# INITIALIZE LOG FILES
# --------------------------------------------------

def initialize_csv(path, headers):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()


initialize_csv(
    PREDICTION_LOG,
    ["prediction_id", "timestamp", "predicted_class", "confidence"]
)

initialize_csv(
    FEEDBACK_LOG,
    ["prediction_id", "actual_class", "is_correct", "timestamp"]
)


def append_csv(path, headers, row):
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writerow(row)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="PlantCare API",
    description="Plant disease classification and monitoring",
    version="2.0.0"
)


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "number_of_classes": len(index_to_class)
    }


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

        image_array = np.asarray(image, dtype=np.float32) / 255.0

        return np.expand_dims(image_array, axis=0)

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image file."
        )


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB.")

    image_array = preprocess_image(image_bytes)

    probabilities = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]

    predictions = []

    for index in top_indices:
        class_name = index_to_class[int(index)]
        plant, disease = class_name.split("___", 1)

        predictions.append({
            "plant": plant,
            "disease": disease,
            "class_name": class_name,
            "confidence": float(probabilities[index])
        })

    prediction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    append_csv(
        PREDICTION_LOG,
        ["prediction_id", "timestamp", "predicted_class", "confidence"],
        {
            "prediction_id": prediction_id,
            "timestamp": timestamp,
            "predicted_class": predictions[0]["class_name"],
            "confidence": predictions[0]["confidence"]
        }
    )

    return {
        "prediction_id": prediction_id,
        "model": MODEL_NAME,
        "predicted_class": predictions[0]["class_name"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions,
        "disclaimer": (
            "Predictions are model estimates, not guaranteed diagnoses."
        )
    }


# --------------------------------------------------
# USER FEEDBACK MODEL
# --------------------------------------------------

class PredictionFeedback(BaseModel):
    prediction_id: str
    actual_class: str


# --------------------------------------------------
# FEEDBACK ENDPOINT
# --------------------------------------------------

@app.post("/feedback")
def submit_feedback(feedback: PredictionFeedback):

    if feedback.actual_class not in class_to_index:
        raise HTTPException(
            status_code=400,
            detail="Unknown class name."
        )

    with open(PREDICTION_LOG, newline="", encoding="utf-8") as file:
        predictions = list(csv.DictReader(file))

    matching = [
        row for row in predictions
        if row["prediction_id"] == feedback.prediction_id
    ]

    if not matching:
        raise HTTPException(
            status_code=404,
            detail="Prediction ID not found."
        )

    with open(FEEDBACK_LOG, newline="", encoding="utf-8") as file:
        existing_feedback = list(csv.DictReader(file))

    if any(
        row["prediction_id"] == feedback.prediction_id
        for row in existing_feedback
    ):
        raise HTTPException(
            status_code=409,
            detail="Feedback already submitted for this prediction."
        )

    predicted_class = matching[0]["predicted_class"]
    is_correct = predicted_class == feedback.actual_class

    append_csv(
        FEEDBACK_LOG,
        ["prediction_id", "actual_class", "is_correct", "timestamp"],
        {
            "prediction_id": feedback.prediction_id,
            "actual_class": feedback.actual_class,
            "is_correct": is_correct,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

    return {
        "message": "Feedback recorded successfully.",
        "prediction_id": feedback.prediction_id,
        "is_correct": is_correct
    }


# --------------------------------------------------
# MONITORING METRICS ENDPOINT
# --------------------------------------------------

@app.get("/metrics")
def get_metrics():

    with open(PREDICTION_LOG, newline="", encoding="utf-8") as file:
        predictions = list(csv.DictReader(file))

    with open(FEEDBACK_LOG, newline="", encoding="utf-8") as file:
        feedback = list(csv.DictReader(file))

    total_predictions = len(predictions)
    labeled_predictions = len(feedback)

    correct_predictions = sum(
        row["is_correct"].lower() == "true"
        for row in feedback
    )

    monitored_accuracy = (
        correct_predictions / labeled_predictions
        if labeled_predictions > 0 else None
    )

    average_confidence = (
        sum(float(row["confidence"]) for row in predictions)
        / total_predictions
        if total_predictions > 0 else None
    )

    return {
        "api_status": "healthy",
        "model_name": MODEL_NAME,
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "total_predictions": total_predictions,
        "labeled_predictions": labeled_predictions,
        "correct_predictions": correct_predictions,
        "monitored_accuracy": monitored_accuracy,
        "average_confidence": average_confidence
    }



# --------------------------------------------------
# RETRAINING CONFIGURATION
# --------------------------------------------------

MIN_LABELLED_SAMPLES = 20
ACCURACY_THRESHOLD = 0.90

RETRAINING_LOG = os.path.join(
    MONITORING_DIR, "retraining_requests.csv"
)

RETRAINING_HEADERS = [
    "request_id",
    "timestamp",
    "reason",
    "labelled_predictions",
    "monitored_accuracy",
    "status"
]

initialize_csv(RETRAINING_LOG, RETRAINING_HEADERS)


# --------------------------------------------------
# RETRAINING ELIGIBILITY
# --------------------------------------------------

def evaluate_retraining_eligibility():

    with open(FEEDBACK_LOG, newline="", encoding="utf-8") as file:
        feedback_records = list(csv.DictReader(file))

    labelled_predictions = len(feedback_records)

    correct_predictions = sum(
        row["is_correct"].lower() == "true"
        for row in feedback_records
    )

    accuracy = (
        correct_predictions / labelled_predictions
        if labelled_predictions > 0 else None
    )

    if labelled_predictions < MIN_LABELLED_SAMPLES:
        eligible = False
        decision = "INSUFFICIENT_DATA"
        reason = (
            f"Only {labelled_predictions} labelled predictions available. "
            f"At least {MIN_LABELLED_SAMPLES} are required."
        )

    elif accuracy < ACCURACY_THRESHOLD:
        eligible = True
        decision = "RETRAINING_RECOMMENDED"
        reason = (
            f"Monitored accuracy ({accuracy:.2%}) is below "
            f"the threshold ({ACCURACY_THRESHOLD:.0%})."
        )

    else:
        eligible = False
        decision = "PERFORMANCE_ACCEPTABLE"
        reason = (
            f"Monitored accuracy ({accuracy:.2%}) meets or exceeds "
            f"the threshold ({ACCURACY_THRESHOLD:.0%})."
        )

    return {
        "eligible": eligible,
        "decision": decision,
        "reason": reason,
        "labelled_predictions": labelled_predictions,
        "correct_predictions": correct_predictions,
        "monitored_accuracy": accuracy,
        "minimum_labelled_samples": MIN_LABELLED_SAMPLES,
        "accuracy_threshold": ACCURACY_THRESHOLD
    }


# --------------------------------------------------
# RETRAINING CHECK ENDPOINT
# --------------------------------------------------

@app.get("/retrain-check")
def retrain_check():

    return evaluate_retraining_eligibility()


# --------------------------------------------------
# RETRAINING REQUEST ENDPOINT
# --------------------------------------------------

@app.post("/retrain")
def request_retraining():

    eligibility = evaluate_retraining_eligibility()

    if not eligibility["eligible"]:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Retraining request not accepted.",
                "decision": eligibility["decision"],
                "reason": eligibility["reason"],
                "labelled_predictions": eligibility["labelled_predictions"]
            }
        )

    # Prevent multiple pending retraining requests
    with open(RETRAINING_LOG, newline="", encoding="utf-8") as file:
        existing_requests = list(csv.DictReader(file))

    if any(row["status"] == "queued" for row in existing_requests):
        raise HTTPException(
            status_code=409,
            detail="A retraining request is already pending."
        )

    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    append_csv(
        RETRAINING_LOG,
        RETRAINING_HEADERS,
        {
            "request_id": request_id,
            "timestamp": timestamp,
            "reason": eligibility["reason"],
            "labelled_predictions": eligibility["labelled_predictions"],
            "monitored_accuracy": eligibility["monitored_accuracy"],
            "status": "queued"
        }
    )

    return {
        "message": "Retraining request successfully queued.",
        "request_id": request_id,
        "status": "queued",
        "note": (
            "This request records the need for retraining. "
            "Model training must be executed separately in the Colab notebook."
        ),
        "eligibility": eligibility
    }

