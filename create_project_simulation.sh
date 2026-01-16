#!/bin/bash

# ==============================================================================
#   Precision Livestock IoT Project - SIMULATION-ONLY Setup Script
# ==============================================================================
# This script sets up the entire project without any hardware requirements.
# It creates a Python script to simulate a sensor device, allowing for
# full backend and data pipeline testing on a single machine.
# ==============================================================================

set -e

print_color() {
    case "$1" in
        "green") echo -e "\n\e[32m$2\e[0m" ;;
        "blue") echo -e "\n\e[34m$2\e[0m" ;;
        "red") echo -e "\n\e[31m$2\e[0m" ;;
        "yellow") echo -e "\n\e[33m$2\e[0m" ;;
        *) echo "$2" ;;
    esac
}

robust_docker_cleanup() {
    print_color "yellow" "Performing a robust cleanup of any old project containers..."
    CONTAINERS_TO_CLEAN="mosquitto influxdb grafana"
    docker stop $CONTAINERS_TO_CLEAN || true
    docker rm $CONTAINERS_TO_CLEAN || true
    echo "✅ Docker environment is clean."
}

main() {
    print_color "green" "=================================================="
    print_color "green" "  Setting up Simulation-Only IoT Project Locally  "
    print_color "green" "=================================================="
    
    robust_docker_cleanup

    local PROJECT_NAME="precision-livestock-iot"
    print_color "blue" "Creating project directory '$PROJECT_NAME'..."
    if [ -d "$PROJECT_NAME" ]; then rm -rf "$PROJECT_NAME"; fi
    mkdir "$PROJECT_NAME" && cd "$PROJECT_NAME"

    print_color "blue" "Creating project structure (no firmware folder)..."
    mkdir -p data/raw data/processed docs notebooks reports/figures results/checkpoints results/metrics scripts src/data src/models src/train src/utils tests

    print_color "blue" "Writing configuration and source code files..."
    echo "venv/" > .gitignore && echo "__pycache__/" >> .gitignore
    echo -e "numpy\npandas\nscipy\nscikit-learn\npaho-mqtt\ninfluxdb-client\nmatplotlib\nseaborn\njupyterlab\nnotebook\ntensorboard" > requirements.txt
    
    cat <<EOL > docker-compose.yml
services:
  mosquitto: {image: eclipse-mosquitto:2.0, container_name: mosquitto, ports: ["1883:1883", "9001:9001"], volumes: ['./docker-data/mosquitto/config:/mosquitto/config', './docker-data/mosquitto/data:/mosquitto/data', './docker-data/mosquitto/log:/mosquitto/log'], restart: unless-stopped}
  influxdb: {image: influxdb:2.7, container_name: influxdb, ports: ["8086:8086"], volumes: ['./docker-data/influxdb/data:/var/lib/influxdb2', './docker-data/influxdb/config:/etc/influxdb2'], restart: unless-stopped, environment: {DOCKER_INFLUXDB_INIT_MODE: setup, DOCKER_INFLUXDB_INIT_USERNAME: admin, DOCKER_INFLUXDB_INIT_PASSWORD: password1234, DOCKER_INFLUXDB_INIT_ORG: farm-org, DOCKER_INFLUXDB_INIT_BUCKET: animal-data}}
  grafana: {image: grafana/grafana:9.5.3, container_name: grafana, ports: ["3000:3000"], volumes: ['./docker-data/grafana/data:/var/lib/grafana'], restart: unless-stopped, depends_on: [influxdb]}
EOL
    mkdir -p docker-data/mosquitto/config && echo -e "persistence true\npersistence_location /mosquitto/data/\nlog_dest file /mosquitto/log/mosquitto.log\nallow_anonymous true" > docker-data/mosquitto/config/mosquitto.conf
    
    # !!!!!!!!!!!!!!! NEW SIMULATOR SCRIPT IS CREATED HERE !!!!!!!!!!!!!!!
    cat <<'EOL' > src/data/sensor_simulator.py
import paho.mqtt.client as mqtt
import time
import json
import random

# --- Configuration ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
# This topic must match what the consumer is listening to
COW_ID = "cow_01"
MQTT_TOPIC = f"farm/cow/{COW_ID}/sensors"

# --- Main Simulation Logic ---
def run_simulator():
    """Connects to MQTT and continuously sends simulated sensor data."""
    client = mqtt.Client(client_id=f"simulator-{COW_ID}")
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT)
        print(f"Simulator connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"Error: Could not connect to MQTT Broker. Is it running? Details: {e}")
        return

    client.loop_start() # Handles reconnects automatically
    
    print(f"Starting to publish data to topic: {MQTT_TOPIC}")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            # Simulate a baseline of gravity on the Z-axis with some noise
            ax = random.uniform(-0.5, 0.5)
            ay = random.uniform(-0.5, 0.5)
            az = 9.81 + random.uniform(-0.2, 0.2)
            
            # Create the data payload
            data = {
                "cow_id": COW_ID,
                "ax": round(ax, 2),
                "ay": round(ay, 2),
                "az": round(az, 2)
            }
            
            # Convert to JSON string
            payload = json.dumps(data)
            
            # Publish to the topic
            result = client.publish(MQTT_TOPIC, payload)
            
            # Optional: Check if publish was successful
            if result.rc == 0:
                print(f"Published: {payload}")
            else:
                print(f"Failed to publish message with result code {result.rc}")

            # Wait for 100ms to simulate 10Hz sampling rate
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nSimulator stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("Simulator disconnected from MQTT Broker.")

if __name__ == "__main__":
    run_simulator()
EOL

    # Python MQTT Consumer Script (remains the same)
    cat <<'EOL' > src/data/mqtt_consumer.py
import os, json; from influxdb_client import InfluxDBClient, Point; from influxdb_client.client.write_api import SYNCHRONOUS; import paho.mqtt.client as mqtt
MQTT_BROKER, MQTT_PORT, MQTT_TOPIC = "localhost", 1883, "farm/cow/+/sensors"
INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET = "http://localhost:8086", "password1234", "farm-org", "animal-data"
influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG); write_api = influx_client.write_api(write_options=SYNCHRONOUS)
def on_connect(c,ud,f,rc):
    if rc==0: c.subscribe(MQTT_TOPIC); print(f"Consumer connected and subscribed to {MQTT_TOPIC}")
    else: print(f"Connection failed, rc={rc}")
def on_message(c,ud,msg):
    try:
        p, cid = msg.payload.decode(), msg.topic.split('/')[2]; print(f"Consumer Rx from {cid}: {p}")
        d = json.loads(p)
        pt = Point("sensor_reading").tag("cow_id", cid).field("ax",d.get("ax",0.0)).field("ay",d.get("ay",0.0)).field("az",d.get("az",0.0))
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=pt)
    except Exception as e: print(f"Error: {e}")
def main():
    mc = mqtt.Client(client_id="influxdb_consumer"); mc.on_connect, mc.on_message = on_connect, on_message; mc.connect(MQTT_BROKER, MQTT_PORT, 60); mc.loop_forever()
if __name__ == '__main__': main()
EOL
    echo "✅ All project files created."

    print_color "blue" "Creating Python virtual environment..."
    python3 -m venv venv
    
    print_color "blue" "Installing Python libraries..."
    venv/bin/pip install --timeout=600 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    venv/bin/pip install -r requirements.txt
    echo "✅ Python environment is ready."
    
    print_color "blue" "Starting backend services (Docker)..."
    docker compose up -d
    echo "✅ Docker services are running."

    print_color "green" "=================================================="
    print_color "green" "    ✅ SIMULATION SETUP IS COMPLETE! ✅    "
    print_color "green" "=================================================="
    print_color "yellow" "Your project is ready. Follow these new steps to run the simulation:"
    echo -e "
    \e[1m1. Open TWO terminal windows and navigate to this project directory in BOTH:\e[0m
       \e[32mcd $(pwd)\e[0m

    \e[1m2. In BOTH terminals, activate the Python Environment:\e[0m
       \e[32msource venv/bin/activate\e[0m
       (You will see '(venv)' in the prompt of both terminals)

    \e[1m3. In Terminal 1 (The Listener), run the DATA CONSUMER:\e[0m
       \e[32mpython src/data/mqtt_consumer.py\e[0m
       (This terminal will now wait for messages.)

    \e[1m4. In Terminal 2 (The Talker), run the SENSOR SIMULATOR:\e[0m
       \e[32mpython src/data/sensor_simulator.py\e[0m
       (This terminal will start generating and sending data.)

    \e[1m5. Watch the output:\e[0m
       - Terminal 2 will show what it's 'Published'.
       - Terminal 1 will show what it has 'Received'. This confirms the data flow!

    \e[1m6. See Your Data in Grafana:\e[0m
       - Go to \e[34mhttp://localhost:3000\e[0m (admin/admin) and set up your dashboard as before.
    "
}

main
