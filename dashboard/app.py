
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import json
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent
API_URL = "http://127.0.0.1:8000"

MONITORING_DIR = PROJECT_DIR / "reports" / "monitoring"
PREDICTIONS_FILE = MONITORING_DIR / "predictions.csv"
FEEDBACK_FILE = MONITORING_DIR / "feedback.csv"
CLASS_MAPPING_FILE = PROJECT_DIR / "data" / "processed" / "class_mapping.json"

st.set_page_config(
    page_title="PlantCare ML Dashboard",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 PlantCare ML Dashboard")
st.caption("Plant disease detection, model monitoring and performance tracking")

# Load class names
with open(CLASS_MAPPING_FILE, "r") as f:
    class_mapping = json.load(f)

class_names = sorted(class_mapping.keys())

# API health and metrics
st.header("System Monitoring")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    metrics = requests.get(f"{API_URL}/metrics", timeout=5).json()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("API Status", health.get("status", "Unknown"))
    col2.metric("Model", metrics.get("model_name", "Unknown"))
    col3.metric("Total Predictions", metrics.get("total_predictions", 0))
    
    accuracy = metrics.get("monitored_accuracy")
    col4.metric(
        "Monitored Accuracy",
        f"{accuracy * 100:.2f}%" if accuracy is not None else "Awaiting feedback"
    )

    st.caption(
        f"API uptime: {metrics.get('uptime_seconds', 0):.0f} seconds"
    )

except requests.exceptions.RequestException:
    st.error("Cannot connect to the PlantCare API. Please start the API server.")
    st.stop()

st.divider()

# Prediction interface
st.header("🔍 Test Plant Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded leaf image", width=300)

    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                uploaded_file.seek(0)
                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state["last_prediction"] = result

                    st.success("Prediction completed successfully!")
                    st.subheader("Prediction Result")

                    st.write(
                        f"**Predicted disease:** {result['predicted_class']}"
                    )
                    st.write(
                        f"**Confidence:** {result['confidence'] * 100:.2f}%"
                    )
                    st.write(
                        f"**Prediction ID:** `{result['prediction_id']}`"
                    )

                    prediction_df = pd.DataFrame(result["top_predictions"])
                    st.dataframe(prediction_df, use_container_width=True)

                    st.caption(result.get("disclaimer", ""))

                else:
                    st.error(f"Prediction failed: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")

# Feedback interface
if "last_prediction" in st.session_state:
    st.divider()
    st.header("📝 Submit Prediction Feedback")

    result = st.session_state["last_prediction"]

    actual_class = st.selectbox(
        "Select the actual plant disease",
        options=class_names,
        key="actual_class"
    )

    if st.button("Submit Feedback"):
        try:
            response = requests.post(
                f"{API_URL}/feedback",
                json={
                    "prediction_id": result["prediction_id"],
                    "actual_class": actual_class
                },
                timeout=10
            )

            if response.status_code == 200:
                feedback = response.json()

                if feedback["is_correct"]:
                    st.success("Feedback recorded. The prediction was correct.")
                else:
                    st.warning("Feedback recorded. The prediction was incorrect.")

                st.session_state.pop("last_prediction", None)
            else:
                st.error(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Feedback submission failed: {e}")

st.divider()

# Visualizations
st.header("📊 Prediction Analytics")

if PREDICTIONS_FILE.exists():
    predictions = pd.read_csv(PREDICTIONS_FILE)

    if not predictions.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Predictions by Disease")
            counts = predictions["predicted_class"].value_counts().reset_index()
            counts.columns = ["Disease", "Count"]

            fig = px.bar(
                counts,
                x="Disease",
                y="Count",
                title="Distribution of Predicted Diseases"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Prediction Confidence")
            fig = px.histogram(
                predictions,
                x="confidence",
                nbins=20,
                title="Distribution of Model Confidence"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Predictions")
        st.dataframe(
            predictions.tail(10).iloc[::-1],
            use_container_width=True
        )

    else:
        st.info("No predictions have been recorded yet.")
else:
    st.info("Prediction monitoring data is not available yet.")

# Refresh
if st.button("🔄 Refresh Dashboard"):
    st.rerun()

st.caption("PlantCare ML Cycle | Model: MobileNetV2 | 27 plant disease classes")


# --------------------------------------------------
# MODEL RETRAINING MANAGEMENT
# --------------------------------------------------

st.divider()
st.header("🔄 Model Retraining Management")
st.caption("Monitor model performance and manage retraining requests.")

try:
    eligibility_response = requests.get(
        f"{API_URL}/retrain-check",
        timeout=10
    )
    eligibility_response.raise_for_status()
    eligibility = eligibility_response.json()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Labelled Predictions",
        f"{eligibility['labelled_predictions']} / "
        f"{eligibility['minimum_labelled_samples']}"
    )

    accuracy = eligibility.get("monitored_accuracy")

    col2.metric(
        "Monitored Accuracy",
        f"{accuracy * 100:.2f}%" if accuracy is not None
        else "Awaiting feedback"
    )

    col3.metric(
        "Accuracy Threshold",
        f"{eligibility['accuracy_threshold'] * 100:.0f}%"
    )

    decision = eligibility["decision"]

    if decision == "INSUFFICIENT_DATA":
        st.info(eligibility["reason"])
    elif decision == "RETRAINING_RECOMMENDED":
        st.warning(eligibility["reason"])
    else:
        st.success(eligibility["reason"])

    if eligibility["eligible"]:
        if st.button("Request Retraining", type="primary"):
            response = requests.post(
                f"{API_URL}/retrain",
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
                st.write("Request ID:", result["request_id"])
                st.info(result["note"])
            else:
                st.error(f"Request failed: {response.text}")
    else:
        st.button(
            "Request Retraining",
            disabled=True,
            help="Retraining is not currently eligible."
        )

except requests.exceptions.RequestException as e:
    st.error(f"Could not retrieve retraining information: {e}")

