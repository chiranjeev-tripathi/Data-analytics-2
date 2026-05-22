# ==========================================
# REAL-TIME PC HEALTH STREAMLIT APP
# ==========================================

import streamlit as st
import psutil
import pandas as pd
import joblib
import time
import plotly.express as px
from datetime import datetime
import os

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------

st.set_page_config(
    page_title="PC Health Monitoring System",
    page_icon="💻",
    layout="wide"
)

# ------------------------------------------
# LOAD ML MODEL
# ------------------------------------------

model = joblib.load("pc_health_model.pkl")

# ------------------------------------------
# TITLE
# ------------------------------------------

st.title("💻 Real-Time PC Health Monitoring & Prediction System")

st.markdown("---")

# ------------------------------------------
# OUTPUT FILE
# ------------------------------------------

output_file = "realtime_prediction_results.csv"

# Create file if not exists
if not os.path.exists(output_file):

    df = pd.DataFrame(columns=[
        "timestamp",
        "cpu",
        "ram",
        "disk",
        "battery",
        "prediction",
        "recommendation"
    ])

    df.to_csv(output_file, index=False)

# ------------------------------------------
# PLACEHOLDERS
# ------------------------------------------

metric_placeholder = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

# ------------------------------------------
# REAL-TIME LOOP
# ------------------------------------------

while True:

    # --------------------------------------
    # GET LIVE SYSTEM METRICS
    # --------------------------------------

    cpu = psutil.cpu_percent(interval=1)

    ram = psutil.virtual_memory().percent

    disk = psutil.disk_usage('/').percent

    battery = psutil.sensors_battery()

    battery_percent = battery.percent if battery else 100

    # --------------------------------------
    # PREPARE INPUT FOR MODEL
    # --------------------------------------

    input_data = pd.DataFrame([{
        "cpu": cpu,
        "ram": ram,
        "disk": disk,
        "battery": battery_percent
    }])

    # --------------------------------------
    # PREDICTION
    # --------------------------------------

    prediction = model.predict(input_data)[0]

    # --------------------------------------
    # RECOMMENDATION ENGINE
    # --------------------------------------

    recommendation = ""

    if cpu > 85:
        recommendation += "Close background applications. "

    if ram > 85:
        recommendation += "Reduce memory usage or upgrade RAM. "

    if disk > 90:
        recommendation += "Clean unnecessary files. "

    if battery_percent < 30:
        recommendation += "Check battery health. "

    if recommendation == "":
        recommendation = "System operating normally."

    # --------------------------------------
    # STORE RESULT
    # --------------------------------------

    result = {
        "timestamp": datetime.now(),
        "cpu": cpu,
        "ram": ram,
        "disk": disk,
        "battery": battery_percent,
        "prediction": prediction,
        "recommendation": recommendation
    }

    result_df = pd.DataFrame([result])

    result_df.to_csv(
        output_file,
        mode='a',
        header=False,
        index=False
    )

    # --------------------------------------
    # LOAD HISTORY
    # --------------------------------------

    history_df = pd.read_csv(output_file)

    # --------------------------------------
    # KPI METRICS
    # --------------------------------------

    with metric_placeholder.container():

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("CPU Usage", f"{cpu}%")
        col2.metric("RAM Usage", f"{ram}%")
        col3.metric("Disk Usage", f"{disk}%")
        col4.metric("Battery", f"{battery_percent}%")
        col5.metric("Prediction", prediction)

    # --------------------------------------
    # ALERTS
    # --------------------------------------

    if prediction == "Critical":
        st.error("🚨 Critical System Condition Detected!")

    elif prediction == "Warning":
        st.warning("⚠️ Warning: System Under Moderate Load")

    else:
        st.success("✅ System Healthy")

    # --------------------------------------
    # RECOMMENDATION BOX
    # --------------------------------------

    st.info(f"💡 Recommendation: {recommendation}")

    # --------------------------------------
    # CHARTS
    # --------------------------------------

    with chart_placeholder.container():

        st.subheader("📊 Real-Time System Analytics")

        # CPU Chart
        fig_cpu = px.line(
            history_df,
            x="timestamp",
            y="cpu",
            title="CPU Usage Trend"
        )

        st.plotly_chart(fig_cpu, use_container_width=True)

        # RAM Chart
        fig_ram = px.line(
            history_df,
            x="timestamp",
            y="ram",
            title="RAM Usage Trend"
        )

        st.plotly_chart(fig_ram, use_container_width=True)

        # Disk Chart
        fig_disk = px.line(
            history_df,
            x="timestamp",
            y="disk",
            title="Disk Usage Trend"
        )

        st.plotly_chart(fig_disk, use_container_width=True)

    # --------------------------------------
    # TABLE
    # --------------------------------------

    with table_placeholder.container():

        st.subheader("📝 Recent Predictions")

        st.dataframe(
            history_df.tail(10),
            use_container_width=True
        )

    # --------------------------------------
    # WAIT
    # --------------------------------------

    time.sleep(5)

    st.rerun() 