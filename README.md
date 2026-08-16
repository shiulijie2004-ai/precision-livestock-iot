# IoT Application for Tracking Farm Animals and Utilizing Their Biological Data

A Precision Livestock Farming (PLF) prototype for monitoring cattle activity and body temperature using wearable IoT sensors, LoRaWAN communication, real-time data storage and visualization, and an XGBoost-based machine learning pipeline.

The system combines an MPU6050 motion sensor, DS18S20 temperature sensor, LiPo-powered wearable hardware, a RAK7249 LoRaWAN gateway, time-series data processing, Grafana visualization, and machine learning for cattle behaviour classification.

## Project Overview

Continuous livestock monitoring through manual observation is time-consuming and difficult to scale. This project explores an end-to-end IoT and machine learning approach for collecting cattle movement and temperature data and transforming those readings into useful behavioural insights.

The system is designed around four main stages:

1. **Sensing** - Collect movement data using the MPU6050 and temperature readings using the DS18S20.
2. **Wireless Transmission** - Transmit field data over LoRaWAN through a RAK7249 gateway.
3. **Data Processing and Visualization** - Ingest, store, and visualize sensor readings using the backend data pipeline, InfluxDB, and Grafana.
4. **Machine Learning** - Preprocess labelled sensor data and train an XGBoost model for cattle behaviour classification.

## System Architecture

```text
+-----------------------------------+
|      Wearable Sensor Device       |
|                                   |
|  MPU6050 Motion Sensor            |
|  DS18S20 Temperature Sensor       |
|  LiPo Battery                     |
+----------------+------------------+
                 |
                 | LoRaWAN
                 v
+-----------------------------------+
|       RAK7249 LoRaWAN Gateway     |
+----------------+------------------+
                 |
                 v
+-----------------------------------+
|        Backend Data Pipeline      |
|   Data Reception / Processing     |
+----------------+------------------+
                 |
                 v
+-----------------------------------+
|             InfluxDB              |
|        Time-Series Storage        |
+----------------+------------------+
                 |
        +--------+---------+
        |                  |
        v                  v
+------------------+  +------------------------+
|     Grafana      |  |  Machine Learning      |
| Live Monitoring  |  |  Preprocess + XGBoost |
+------------------+  +------------------------+
```

## Key Features

- Wearable cattle monitoring using IoT sensors.
- Motion sensing with the MPU6050 accelerometer and gyroscope.
- Temperature monitoring using the DS18S20 digital temperature sensor.
- LiPo battery-powered wearable operation.
- Long-range wireless communication using LoRaWAN.
- RAK7249 LoRaWAN gateway for receiving field sensor transmissions.
- Time-series data storage using InfluxDB.
- Grafana dashboards for real-time sensor visualization.
- Data preprocessing and feature preparation for machine learning.
- XGBoost-based cattle behaviour classification.
- Model evaluation using classification metrics and confusion matrices.
- Containerized backend services using Docker and Docker Compose where applicable.

## Hardware Components

| Component | Purpose |
|---|---|
| MPU6050 | Measures acceleration and angular velocity for cattle movement analysis |
| DS18S20 | Measures body or surrounding temperature readings |
| LiPo Battery | Provides portable power for the wearable sensor device |
| RAK7249 | Receives LoRaWAN transmissions from field sensor devices and forwards data to the backend |

## Technology Stack

| Layer | Technology |
|---|---|
| Motion Sensing | MPU6050 accelerometer + gyroscope |
| Temperature Sensing | DS18S20 digital temperature sensor |
| Power | LiPo battery |
| Wireless Communication | LoRaWAN |
| LoRaWAN Gateway | RAK7249 |
| Backend Processing | Python |
| Messaging / Data Integration | MQTT where configured in the backend pipeline |
| Time-Series Database | InfluxDB |
| Visualization | Grafana |
| Machine Learning | XGBoost |
| Data Science | Python, pandas, NumPy, scikit-learn |
| Deployment | Docker / Docker Compose |

## Data Flow

```text
Cattle Movement + Temperature
            |
            v
 MPU6050 + DS18S20
            |
            v
  Wearable Sensor Device
      powered by LiPo
            |
            | LoRaWAN
            v
     RAK7249 Gateway
            |
            v
    Backend Data Pipeline
            |
     +------+------+
     |             |
     v             v
 InfluxDB      ML Dataset
     |             |
     v             v
  Grafana       XGBoost
                     |
                     v
          Behaviour Classification
```

## Sensor Data

### MPU6050

The MPU6050 provides six-axis motion measurements:

- `ax` - acceleration on the X-axis
- `ay` - acceleration on the Y-axis
- `az` - acceleration on the Z-axis
- `gx` - angular velocity on the X-axis
- `gy` - angular velocity on the Y-axis
- `gz` - angular velocity on the Z-axis

These readings can be used to identify movement patterns associated with different cattle behaviours.

### DS18S20

The DS18S20 provides digital temperature readings that can be recorded together with movement data. Temperature information adds another biological signal to the livestock monitoring pipeline and can support future health-related analysis.

## Machine Learning Pipeline

The machine learning workflow transforms collected sensor data into labelled samples for cattle behaviour classification.

```text
Raw Sensor Data
      |
      v
Data Cleaning
      |
      v
Segmentation / Windowing
      |
      v
Feature Engineering
      |
      v
Label Assignment
      |
      v
XGBoost Training
      |
      v
Model Evaluation
      |
      v
Behaviour Prediction
```

### XGBoost Model

XGBoost is used as the main machine learning model for cattle behaviour classification. Features generated from the collected sensor readings are used as model inputs, while labelled cattle behaviours are used as the prediction targets.

The model can be evaluated using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Getting Started

### Prerequisites

Depending on the part of the system being run, the project may require:

- Python 3.9+
- Conda or Miniconda
- Docker
- Docker Compose
- Git
- Firmware development environment for the wearable node
- Access to the RAK7249 LoRaWAN gateway configuration

### 1. Clone the Repository

```bash
git clone https://github.com/shiulijie2004-ai/precision-livestock-iot.git
cd precision-livestock-iot
```

### 2. Create the Python Environment

```bash
conda env create -f environment.yml
conda activate plf-iot
```

### 3. Start Backend Services

If the repository uses the included Docker Compose configuration:

```bash
docker-compose up -d
```

Typical local services include:

| Service | Default Address |
|---|---|
| MQTT Broker | `localhost:1883` |
| InfluxDB | `http://localhost:8086` |
| Grafana | `http://localhost:3000` |

### 4. Configure the Wearable Sensor Device

Connect and configure the hardware used by the wearable node:

- MPU6050 motion sensor
- DS18S20 temperature sensor
- LiPo battery
- LoRaWAN-capable communication hardware used by the node

Configure the firmware with the required sensor, LoRaWAN, and device settings before deployment.

### 5. Configure the RAK7249 Gateway

Configure the RAK7249 as the LoRaWAN gateway for receiving data from the wearable device and forwarding it to the backend infrastructure used by the project.

Gateway parameters depend on the LoRaWAN deployment and network configuration used during testing.

### 6. Run the Data Ingestion Pipeline

Start the Python ingestion or consumer component used by the project to receive sensor data and write processed readings to InfluxDB.

For repositories using `mqtt_consumer.py`:

```bash
python src/data/mqtt_consumer.py
```

### 7. Visualize Sensor Data

Open Grafana:

```text
http://localhost:3000
```

Configure InfluxDB as the data source and build dashboard panels for fields such as:

- Acceleration X, Y, Z
- Gyroscope X, Y, Z
- Temperature
- Device or cattle identifier
- Timestamp

## Repository Structure

```text
precision-livestock-iot/
├── data/
│   ├── raw/                 # Raw sensor datasets
│   └── processed/           # Cleaned / processed datasets
├── docker-compose.yml       # Backend service configuration
├── firmware/                # Wearable sensor firmware
├── notebooks/               # EDA and ML experiments
├── reports/                 # Figures and project outputs
├── results/                 # Model metrics and saved outputs
├── src/
│   ├── data/                # Data ingestion and preprocessing
│   ├── models/              # XGBoost model training / evaluation
│   └── utils/               # Shared utilities
├── tests/                   # Testing and validation
└── README.md
```

## Project Status

The project focuses on integrating the following components into a complete Precision Livestock Farming prototype:

- MPU6050-based cattle movement sensing.
- DS18S20-based temperature sensing.
- LiPo-powered wearable operation.
- LoRaWAN communication.
- RAK7249 gateway connectivity.
- Backend sensor data ingestion.
- InfluxDB time-series storage.
- Grafana dashboard visualization.
- Sensor data preprocessing and feature engineering.
- XGBoost behaviour classification.

## Future Improvements

Potential improvements include:

- Collecting a larger real-world cattle dataset.
- Increasing the number of labelled cattle behaviour classes.
- Improving wearable enclosure durability for farm deployment.
- Optimizing LiPo battery consumption and operating time.
- Improving LoRaWAN transmission reliability and coverage.
- Adding real-time XGBoost inference to the live data pipeline.
- Displaying predicted behaviours directly in Grafana.
- Adding automated alerts for unusual movement or temperature patterns.
- Exploring additional biological sensors for livestock health monitoring.

## Project Purpose

This project demonstrates how IoT sensing, LoRaWAN communication, time-series data engineering, real-time visualization, and machine learning can be integrated into a single Precision Livestock Farming prototype for cattle monitoring.

The system is intended for academic and experimental use and provides a foundation for future development of scalable livestock monitoring and behaviour analysis solutions.
