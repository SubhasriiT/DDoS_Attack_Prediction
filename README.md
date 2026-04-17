# AI-Based DDoS Attack Prediction and Prevention Using Network Behavior Modeling

## 🚀 Live Demo

👉 Try the app here:  
https://ddosattackprediction-7xbdduef9rordymwyyhzov.streamlit.app/

An AI + Deep Learning project that detects DDoS attacks at early stages by analyzing
temporal network traffic behavior using a Hybrid Hierarchical BiGRU model and a
real-time Streamlit monitoring dashboard.

## Description

Traditional DDoS detection systems classify traffic as simply normal or attack — often
too late to prevent damage. This project introduces a **three-stage detection approach**:

| Stage | Label | Meaning |
|---|---|---|
| 0 | 🟢 Normal | Regular network traffic |
| 1 | 🟡 Early DDoS Warning | Suspicious patterns detected early |
| 2 | 🔴 Active Attack | Full DDoS attack in progress |

By detecting attacks before they fully develop, the system enables **proactive defense**
rather than reactive response.

## Project Workflow

### Step 1 — Data Preparation
- Loaded 4 training + 4 testing parquet files from the CICDDoS2019 dataset
- Removed duplicates, missing values, and infinite values
- Selected 41 important network traffic features (packet rates, flow duration, TCP flags)
- Created custom **Early Stage labels** from the first portion of each attack sequence

### Step 2 — Exploratory Data Analysis
- Attack type distribution analysis
- Normal vs attack traffic comparison
- Feature correlation heatmaps and pairplots
- Confirmed clear behavioral differences between normal and attack traffic

### Step 3 — Models Implemented

| Model | Approach | Accuracy |
|---|---|---|
| Gradient Boosting | Baseline ML classifier | 68% |
| GRU | Temporal deep learning (40-packet sequences) | 75% |
| **Hybrid Hierarchical BiGRU** | Autoencoder + two-stage BiGRU | 75% |

### Step 4 — Hybrid Model Architecture
The best-performing model works in two stages:
1. **Stage 1 Model** — Binary classifier: Normal vs DDoS
2. **Stage 2 Model** — Multiclass: Early DDoS vs Full Attack

An **Autoencoder** trained on normal traffic extracts anomaly features, which are
combined with original features for richer input representation.

### Step 5 — Streamlit Dashboard
A real-time Security Operations Center (SOC) interface providing:
- **Dataset Analysis Mode** — Upload and analyze complete traffic logs
- **Live Traffic Simulation Mode** — Stream traffic window by window
- Automated mitigation: allow / rate-limit / block IP
- Live SOC metrics: blocked IPs, rate-limited IPs, total attacks
- Real-time detection trend graph

## Technologies Used

- **Language:** Python
- **Environment:** Jupyter Notebook, Streamlit
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, keras, streamlit, pyarrow
- **Models:** Gradient Boosting, GRU, Autoencoder, Hierarchical BiGRU
- **Dataset:** CICDDoS2019 (Kaggle)

## Dataset

This project uses the **CICDDoS2019** dataset available on Kaggle:
> 🔗 [CICDDoS2019 Dataset — Kaggle](https://www.kaggle.com/datasets/dhoogla/cicddos2019)

Only the following 8 files were used:

**Training files:**
- `Syn-training.parquet`
- `UDP-training.parquet`
- `LDAP-training.parquet`
- `MSSQL-training.parquet`

**Testing files:**
- `Syn-testing.parquet`
- `UDP-testing.parquet`
- `LDAP-testing.parquet`
- `MSSQL-testing.parquet`

> ⚠️ The raw parquet files are not included in this repository due to their large size.
> Download them from the Kaggle link above and place them in a `raw_data/` folder
> before running the notebook.

## Installation

1. Clone the repository:
git clone https://github.com/SubhasriiT/DDoS_Attack_Prediction.git

2. Navigate to the project folder:
cd ddos-attack-prediction

3. Install required libraries:
pip install -r requirements.txt

## Usage

### Running the Jupyter Notebook
Open the notebook to explore the full ML pipeline:
jupyter notebook DDoS_Analysis.ipynb

Run all cells in order to perform data preprocessing, EDA, model training, and evaluation.

### Running the Streamlit Dashboard
streamlit run app.py

Once launched:
1. Choose **Dataset Analysis** or **Live Traffic Simulation** from the sidebar
2. Upload `DDoS_Processed_Data.csv` from the `Data/` folder
3. Click **Start Monitoring** to begin live simulation
4. Observe real-time classifications, SOC metrics, and automated mitigation actions

## Results

| Metric | Value |
|---|---|
| Best Model | Hybrid Hierarchical BiGRU |
| Best Accuracy | 75% |
| Detection Stages | Normal / Early DDoS / Full Attack |
| Mitigation Actions | Allow / Rate-Limit / Block IP |
| Dashboard | Real-time Streamlit SOC Interface |

## Key Features

- Three-stage detection: Normal → Early Warning → Full Attack
- Temporal behavior modeling using GRU and BiGRU
- Anomaly detection using Autoencoder on normal traffic
- Hierarchical classification for precise attack staging
- Real-time Streamlit SOC dashboard with live metrics
- Automated IP mitigation (rate-limit and block)
- Works on standard network traffic CSV data

## 💡 Why This Project Matters

- Detects DDoS attacks *before they fully occur*
- Uses **temporal deep learning (BiGRU)** for sequence modeling
- Implements **hierarchical classification** for better accuracy
- Combines **autoencoder + supervised learning**
- Provides a **real-time SOC dashboard**

## ⚠️ Known Issues

- Model loading may fail if TensorFlow/Keras versions mismatch
- Ensure correct versions are installed using `requirements.txt`
- Large datasets may slow down live simulation mode

## Contributing

Contributions are welcome. If you would like to improve this project:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the
[MIT License](https://opensource.org/licenses/MIT)
and is intended for educational purposes.

## Contact

For any queries or suggestions, feel free to reach out:

**Name:** Subhasri

**Email:** [your-email@gmail.com]

**GitHub:** [SubhasriiT](https://github.com/SubhasriiT)
