import streamlit as st
import requests

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Industrial Predictive Maintenance")
st.write(
    "Enter machine sensor readings to predict the risk of machine failure."
)

st.divider()

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=0.1,
    value=300.0
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=0.1,
    value=310.0
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=1,
    value=1500,
    step=1
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    value=100,
    step=1
)

st.divider()

if st.button(
    "Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    payload = {
        "type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            probability = result["failure_probability"]
            prediction = result["prediction"]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("⚠️ Machine Failure Risk Detected")
            else:
                st.success("✅ No Failure Predicted")

            st.metric(
                "Failure Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(
                min(float(probability), 1.0)
            )

            st.caption(
                f"Decision threshold: "
                f"{result['threshold']}"
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.json(response.json())

    except requests.exceptions.RequestException:

        st.error(
            "Cannot connect to the prediction API. "
            "Make sure the FastAPI Docker container is running."
        )