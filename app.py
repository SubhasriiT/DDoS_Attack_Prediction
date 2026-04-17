import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time

# Safe winsound (Cloud compatible)
try:
    import winsound
    WINSOUND_AVAILABLE = True
except:
    WINSOUND_AVAILABLE = False

# ================================
# CONFIGURATION
# ================================
st.set_page_config(page_title="AI-Driven DDoS Early Warning System", layout="wide")

WINDOW_SIZE = 30
label_map = {0: "Normal", 1: "Early_DDoS", 2: "Attack"}

# ================================
# LOAD MODEL ASSETS
# ================================
@st.cache_resource
def load_assets():
    try:
        stage1_model = tf.keras.models.load_model("stage1.h5", compile=False)
        stage2_model = tf.keras.models.load_model("stage2.h5", compile=False)
        encoder = tf.keras.models.load_model("encoder.h5", compile=False)

        scaler = joblib.load("scaler.pkl")
        scaler_encoded = joblib.load("scaler_encoded.pkl")

        return stage1_model, stage2_model, encoder, scaler, scaler_encoded

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
stage1_model, stage2_model, encoder, scaler, scaler_encoded = load_assets()
# ================================
# FEATURE LIST
# ================================

traffic_features = [
    'Flow Packets/s','Flow Bytes/s','Total Fwd Packets','Total Backward Packets',
    'Fwd Packets/s','Bwd Packets/s','Down/Up Ratio'
]

duration_features = [
    'Flow Duration','Active Mean','Active Std','Active Max','Active Min',
    'Idle Mean','Idle Std','Idle Max','Idle Min'
]

iat_features = [
    'Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min',
    'Fwd IAT Mean','Fwd IAT Std','Bwd IAT Mean','Bwd IAT Std'
]

tcp_flag_features = [
    'SYN Flag Count','ACK Flag Count','RST Flag Count','PSH Flag Count','FIN Flag Count'
]

packet_features = [
    'Packet Length Mean','Packet Length Std','Packet Length Variance',
    'Packet Length Min','Packet Length Max','Avg Packet Size',
    'Avg Fwd Segment Size','Avg Bwd Segment Size'
]

subflow_features = [
    'Subflow Fwd Packets','Subflow Fwd Bytes',
    'Subflow Bwd Packets','Subflow Bwd Bytes'
]

selected_features = (
    traffic_features + duration_features + iat_features +
    tcp_flag_features + packet_features + subflow_features
)

# ================================
# UI HEADER
# ================================

st.title("🛡 AI-Driven DDoS Early Warning & Mitigation System")
st.markdown("### Hybrid Hierarchical Deep Learning Model for Proactive Attack Detection")
st.markdown("Traffic Class: **0 = Normal | 1 = Early Warning | 2 = Attack**")
st.markdown("---")

mode = st.sidebar.radio(
    "Detection Mode",
    ["Upload Dataset", "Live Traffic Simulation"]
)

# =========================================================
# MODE 1 — DATASET ANALYSIS
# =========================================================

if mode == "Upload Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload Network Traffic CSV", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### 📄 Uploaded Data Preview", data.head())

        missing_cols = [col for col in selected_features if col not in data.columns]

        if len(missing_cols) > 0:
            st.error(f"Missing Required Columns: {missing_cols}")
        else:
            X_raw = data[selected_features].copy()
            X_scaled = scaler.transform(X_raw)
            X_scaled = np.array(X_scaled).astype(np.float32)

            X_seq = []
            for i in range(len(X_scaled) - WINDOW_SIZE):
                X_seq.append(X_scaled[i:i+WINDOW_SIZE])

            X_seq = np.array(X_seq)

            if len(X_seq) == 0:
                st.error(f"Need at least {WINDOW_SIZE+1} rows for prediction.")
            else:
                with st.spinner("🔍 Analyzing Network Traffic..."):
                    n_samples, timesteps, n_features = X_seq.shape
                    X_reshaped = X_seq.reshape(-1, n_features)

                    encoded_features = encoder.predict(X_reshaped, verbose=0)
                    encoded_scaled = scaler_encoded.transform(encoded_features)
                    encoded_seq = encoded_scaled.reshape(n_samples, timesteps, -1)

                    X_combined = np.concatenate([X_seq, encoded_seq], axis=2)

                    stage1_probs = stage1_model.predict(X_combined, verbose=0)
                    stage1_labels = (stage1_probs > 0.5).astype(int).flatten()

                    pred_labels = []

                    for i in range(len(stage1_labels)):
                        if stage1_labels[i] == 0:
                            pred_labels.append(0)
                        else:
                            stage2_prob = stage2_model.predict(X_combined[i:i+1], verbose=0)
                            stage2_label = np.argmax(stage2_prob)
                            pred_labels.append(stage2_label + 1)

                    # ================================
                    # Traffic Stage Distribution
                    # ================================
                    st.markdown("### 📈 Traffic Stage Distribution Across Dataset")

                    stage_counts = pd.Series(pred_labels).value_counts().sort_index()

                    normal_count = stage_counts.get(0, 0)
                    early_count = stage_counts.get(1, 0)
                    attack_count_dist = stage_counts.get(2, 0)
                    dist_df = pd.DataFrame({
                        "Traffic Stage": ["Normal", "Early_DDoS", "Attack"],
                        "Count": [normal_count, early_count, attack_count_dist]
                    })
                    st.bar_chart(dist_df.set_index("Traffic Stage"))

                final_prediction = pred_labels[-1]

                if final_prediction == 0:
                    st.success("🟢 NORMAL TRAFFIC – System Operating Normally")
                elif final_prediction == 1:
                    st.warning("🟡 EARLY DDoS WARNING – Suspicious Activity Detected")
                else:
                    st.error("🔴 ACTIVE DDoS ATTACK DETECTED")

                # ================================
                # SIMULATED MITIGATION ANALYSIS
                # ================================
                blocked_ips = []
                rate_limited_ips = []
                attack_count = 0

                for pred in pred_labels:
                    src_ip = f"192.168.1.{np.random.randint(2,200)}"
                    if pred == 1:
                        if src_ip not in rate_limited_ips and src_ip not in blocked_ips:
                            rate_limited_ips.append(src_ip)
                    elif pred == 2:
                        attack_count += 1
                        if src_ip not in blocked_ips:
                            blocked_ips.append(src_ip)
                        if src_ip in rate_limited_ips:
                            rate_limited_ips.remove(src_ip)

                # ================================
                # SOC METRICS
                # ================================
                st.markdown("### 📊 Security Operations Center Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("🚫 Blocked IPs", len(blocked_ips))
                col2.metric("⚠ Rate Limited IPs", len(rate_limited_ips))
                col3.metric("🔥 Total Attacks Detected", attack_count)

                # ================================
                # MITIGATION TABLES
                # ================================
                st.markdown("### 🚧 Active Mitigation Actions")
                colA, colB = st.columns(2)

                with colA:
                    st.markdown("#### ⚠ Rate Limited IPs")
                    if len(rate_limited_ips) == 0:
                        st.info("No IPs were rate limited during analysis.")
                    else:
                        rate_view = st.selectbox(
                            "View Rate Limited IPs",
                            ["Latest 20", "Show All"],
                            key="dataset_rate_view"
                        )
                        rate_data = rate_limited_ips[-20:] if rate_view == "Latest 20" else rate_limited_ips
                        st.table(pd.DataFrame(rate_data, columns=["IP Address"]))

                with colB:
                    st.markdown("#### 🚫 Blocked IPs")
                    if len(blocked_ips) == 0:
                        st.info("No IPs were blocked during analysis.")
                    else:
                        block_view = st.selectbox(
                            "View Blocked IPs",
                            ["Latest 20", "Show All"],
                            key="dataset_block_view"
                        )
                        block_data = blocked_ips[-20:] if block_view == "Latest 20" else blocked_ips
                        st.table(pd.DataFrame(block_data, columns=["IP Address"]))

# =========================================================
# MODE 2 — LIVE TRAFFIC SIMULATION
# =========================================================

elif mode == "Live Traffic Simulation":
    st.sidebar.markdown("### Live Stream Settings")

    stream_file = st.sidebar.file_uploader("Upload Traffic Log for Streaming", type=["csv"])
    stream_speed = st.sidebar.slider("Stream Speed (seconds)", 0.05, 2.0, 0.3)

    if "running" not in st.session_state:
        st.session_state.running = False
    if "blocked_ips" not in st.session_state:
        st.session_state.blocked_ips = []
    if "rate_limited_ips" not in st.session_state:
        st.session_state.rate_limited_ips = []
    if "previous_state" not in st.session_state:
        st.session_state.previous_state = 0
    if "attack_count" not in st.session_state:
        st.session_state.attack_count = 0
    if "attack_alert_shown" not in st.session_state:
        st.session_state.attack_alert_shown = False

    start_button = st.sidebar.button("▶ Start Monitoring")
    stop_button = st.sidebar.button("⛔ Stop Monitoring")

    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False

    if stream_file:
        data = pd.read_csv(stream_file)
        st.write("### 📄 Stream Source Preview")
        st.dataframe(data.head())

        st.markdown("### 📊 Security Operations Center Metrics")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        blocked_metric = metric_col1.empty()
        rate_metric = metric_col2.empty()
        attack_metric = metric_col3.empty()

        blocked_metric.metric("🚫 Blocked IPs", len(st.session_state.blocked_ips))
        rate_metric.metric("⚠ Rate Limited IPs", len(st.session_state.rate_limited_ips))
        attack_metric.metric("🔥 Total Attacks Detected", st.session_state.attack_count)

        chart_placeholder = st.empty()
        prediction_placeholder = st.empty()
        action_placeholder = st.empty()
        ip_placeholder = st.empty()

        attack_alert_placeholder = st.empty()
        st.markdown("### 🚧 Active Mitigation Actions")
        table_col1, table_col2 = st.columns(2)

        with table_col1:
            st.markdown("#### ⚠ Rate Limited IPs")
            rate_limit_placeholder = st.empty()
            if not st.session_state.running and len(st.session_state.rate_limited_ips) > 0:
                rate_view = st.selectbox("View Rate Limited IPs", ["Latest 20", "Show All"], key="rate_view")
                rate_data = st.session_state.rate_limited_ips[-20:] if rate_view == "Latest 20" else st.session_state.rate_limited_ips
                rate_limit_placeholder.table(pd.DataFrame(rate_data, columns=["IP Address"]))
            elif len(st.session_state.rate_limited_ips) > 0:
                rate_limit_placeholder.table(pd.DataFrame(st.session_state.rate_limited_ips, columns=["IP Address"]))

        with table_col2:
            st.markdown("#### 🚫 Blocked IPs")
            blocked_placeholder = st.empty()
            if not st.session_state.running and len(st.session_state.blocked_ips) > 0:
                block_view = st.selectbox("View Blocked IPs", ["Latest 20", "Show All"], key="block_view")
                block_data = st.session_state.blocked_ips[-20:] if block_view == "Latest 20" else st.session_state.blocked_ips
                blocked_placeholder.table(pd.DataFrame(block_data, columns=["IP Address"]))
            elif len(st.session_state.blocked_ips) > 0:
                blocked_placeholder.table(pd.DataFrame(st.session_state.blocked_ips, columns=["IP Address"]))

        # Pre-processing for live loop
        X_raw = data[selected_features].copy()
        X_scaled = scaler.transform(X_raw)
        X_scaled = np.array(X_scaled).astype(np.float32)
        start_index = 1950
        trend_df = pd.DataFrame(columns=["Predicted_Class"])

        if st.session_state.running:
            st.success("▶ Live Monitoring Started")
            for i in range(start_index + WINDOW_SIZE, len(X_scaled)):
                if not st.session_state.running:
                    break

                window = X_scaled[i-WINDOW_SIZE:i].reshape(1, WINDOW_SIZE, -1)
                reshaped = window.reshape(-1, window.shape[2])

                encoded = encoder.predict(reshaped, verbose=0)
                encoded_scaled = scaler_encoded.transform(encoded)
                encoded_seq = encoded_scaled.reshape(1, WINDOW_SIZE, -1)
                combined = np.concatenate([window, encoded_seq], axis=2)

                stage1_prob = stage1_model.predict(combined, verbose=0)[0][0]
                if stage1_prob <= 0.5:
                    pred = 0
                else:
                    stage2_prob = stage2_model.predict(combined, verbose=0)
                    pred = np.argmax(stage2_prob) + 1

                src_ip = f"192.168.1.{np.random.randint(2,200)}"
                timestamp = time.strftime("%H:%M:%S")

                prediction_placeholder.markdown(f"### Current Prediction: **{label_map[pred]}**\n\n🕒 **Time:** {timestamp}\n\n🌐 **Source IP:** {src_ip}")

                if pred == 2 and not st.session_state.attack_alert_shown:
                    attack_alert_placeholder.error("🚨 Attack Phase Detected – Immediate Blocking Activated")
                    st.session_state.attack_alert_shown = True

                if pred == 0:
                    action_placeholder.success("✅ Traffic Allowed")
                elif pred == 1:
                    action_placeholder.warning("⚠ Rate Limiting Applied")
                    if src_ip not in st.session_state.rate_limited_ips and src_ip not in st.session_state.blocked_ips:
                        st.session_state.rate_limited_ips.append(src_ip)
                else:
                    action_placeholder.error(f"⛔ IP BLOCKED: {src_ip}")
                    if src_ip not in st.session_state.blocked_ips:
                        st.session_state.blocked_ips.append(src_ip)
                    if src_ip in st.session_state.rate_limited_ips:
                        st.session_state.rate_limited_ips.remove(src_ip)

                if pred != st.session_state.previous_state:
                    if WINSOUND_AVAILABLE:
                        if pred == 1:
                            winsound.Beep(1000, 400)
                        elif pred == 2:
                            winsound.Beep(1500, 700)
                        elif pred == 0:
                            winsound.Beep(700, 300)
                    if pred == 2:
                        st.session_state.attack_count += 1

                st.session_state.previous_state = pred

                # Update metrics and tables live
                blocked_metric.metric("🚫 Blocked IPs", len(st.session_state.blocked_ips))
                rate_metric.metric("⚠ Rate Limited IPs", len(st.session_state.rate_limited_ips))
                attack_metric.metric("🔥 Total Attacks Detected", st.session_state.attack_count)

                rate_limit_placeholder.table(pd.DataFrame(st.session_state.rate_limited_ips, columns=["IP Address"]))
                blocked_placeholder.table(pd.DataFrame(st.session_state.blocked_ips, columns=["IP Address"]))

                trend_df.loc[len(trend_df)] = pred
                chart_placeholder.line_chart(trend_df, use_container_width=True)
                time.sleep(stream_speed)
    else:
        st.info("Upload a CSV file to simulate live network traffic.")
