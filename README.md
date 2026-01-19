# IoT Application for Tracking Farm Animals and Utilizing Their Biological Data

An open-source, low-cost Precision Livestock Farming (PLF) system for **real-time cattle monitoring** using wearable sensors, an on-premise gateway, and a machine learning pipeline to detect key behaviours and potential health anomalies (e.g., lameness/estrus). :contentReference[oaicite:3]{index=3}

---

## About the Project

Modern livestock farming faces challenges in monitoring animal health and behaviour efficiently. This project builds an end-to-end IoT + ML prototype that:
- Collects biological/behavioural signals from **wearable sensor nodes** (e.g., accelerometer/IMU, temperature; optionally GPS).
- Streams data to an on-farm **gateway** (e.g., Raspberry Pi).
- Ingests via **MQTT**, stores time-series data in **InfluxDB**, and visualizes insights in **Grafana**.
- Trains ML models (e.g., **LSTM**) to classify behaviours and support anomaly detection. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

### Target Users
Small-to-medium cattle/dairy farms, agricultural researchers, and open-source hardware enthusiasts. :contentReference[oaicite:6]{index=6}

---
Getting Started
Prerequisites

Python 3.9+ 


Conda (recommended) 


Docker + Docker Compose 


Git


1) Clone the Repository

git clone https://github.com/shiulijie2004-ai/precision-livestock-iot.git

cd precision-livestock-iot


2) Create the Python Environment (Conda)

conda env create -f environment.yml

conda activate plf-iot



3) Launch Backend Services (Mosquitto + InfluxDB + Grafana)

docker-compose up -d


Service URLs/Ports: 


MQTT Broker: localhost:1883

InfluxDB UI: http://localhost:8086

Grafana: http://localhost:3000


Usage
A) Send Sensor Data (MQTT)

Configure your ESP32 firmware in firmware/ to publish sensor readings to the MQTT broker.

Use topic conventions like:

farm/<cow_id>/imu

farm/<cow_id>/temp

A Python ingestion script (in src/data/) can subscribe and write into InfluxDB.

Add your final topic schema and message format here once finalized.


B) Visualize in Grafana

Open Grafana: http://localhost:3000

Add InfluxDB as a data source.

Import or build dashboards (e.g., activity timeline, temperature trends, anomaly flags).


C) Train & Evaluate Models

Place datasets in data/raw/

Run preprocessing scripts in src/data/

Train models via scripts in src/train/

Save metrics to results/metrics/ and checkpoints to results/checkpoints/



## Repository Structure

Suggested project structure: :contentReference[oaicite:14]{index=14}

```text
precision-livestock-iot/
├── data/
│   ├── raw/                # raw datasets (not tracked)
│   └── processed/          # processed datasets (not tracked)
├── docker-compose.yml      # backend services (Mosquitto/InfluxDB/Grafana)
├── docker-data/            # persisted volumes for containers (local only)
├── docs/                   # documents, diagrams, writeups
├── firmware/               # ESP32 sensor-node firmware
├── notebooks/              # EDA / experiments
├── reports/figures/        # exported figures
├── results/
│   ├── checkpoints/        # saved training checkpoints
│   └── metrics/            # evaluation outputs
├── scripts/                # helper scripts
├── src/
│   ├── data/               # ingestion + preprocessing
│   ├── models/             # model definitions
│   ├── train/              # training + evaluation
│   └── utils/              # shared utilities
└── tests/                  # unit/integration tests



