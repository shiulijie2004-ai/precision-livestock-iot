import os, json, math
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, deque

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv, find_dotenv

# --- Robust .env loading (prefer find_dotenv; fallback to project root) ---
ENV_PATH = find_dotenv(usecwd=True)
if not ENV_PATH:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parents[1]  # adjust if your file is not under /src/*
    ENV_PATH = str(PROJECT_ROOT / ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

# --- Config ---
MQTT_BROKER = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 1883))

MQTT_SENSORS_TOPIC = os.getenv("MQTT_SENSORS_TOPIC", "farm/cow/+/sensors")
MQTT_PRED_TOPIC    = os.getenv("MQTT_PRED_TOPIC",    "farm/cow/+/predictions")

INFLUXDB_URL    = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN  = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG    = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "animal-data")

MEAS_SENSORS  = os.getenv("MEAS_SENSORS",  "sensor_reading")
MEAS_PRED     = os.getenv("MEAS_PRED",     "predictions")
MEAS_METRICS  = os.getenv("MEAS_METRICS",  "pipeline_metrics")

# ---- Missingness config (simple + light) ----
EXPECTED_INTERVAL_S = float(os.getenv("EXPECTED_PUBLISH_INTERVAL_S", "1.0"))  # simulator publish interval
GAP_FACTOR = float(os.getenv("GAP_FACTOR", "2.0"))  # if gap > GAP_FACTOR * interval => consider missing
METRICS_EVERY_N_MESSAGES = int(os.getenv("METRICS_EVERY_N_MESSAGES", "10"))

# ---- Deviation score config (EWMA-based) ----
# we compute EWMA mean/std for temperature per cow
DEV_ALPHA = float(os.getenv("DEV_ALPHA", "0.02"))  # smaller = smoother baseline
DEV_MIN_STD = float(os.getenv("DEV_MIN_STD", "0.05"))  # avoid divide-by-0
DEV_Z_CLIP = float(os.getenv("DEV_Z_CLIP", "10.0"))  # limit extreme z

masked = (INFLUXDB_TOKEN[:6] + "..." + INFLUXDB_TOKEN[-4:]) if INFLUXDB_TOKEN else None
print("\n[consumer config]")
print(" INFLUXDB_URL    =", INFLUXDB_URL)
print(" INFLUXDB_ORG    =", INFLUXDB_ORG)
print(" INFLUXDB_BUCKET =", INFLUXDB_BUCKET)
print(" INFLUXDB_TOKEN  =", masked)
print(" MQTT_BROKER     =", f"{MQTT_BROKER}:{MQTT_PORT}")
print(" SUB sensors     =", MQTT_SENSORS_TOPIC)
print(" SUB predictions =", MQTT_PRED_TOPIC)
print(" MEAS sensors    =", MEAS_SENSORS)
print(" MEAS predictions=", MEAS_PRED)
print(" MEAS metrics    =", MEAS_METRICS)
print(" EXPECTED_INTERVAL_S =", EXPECTED_INTERVAL_S)
print(" GAP_FACTOR          =", GAP_FACTOR, "\n")

if not all([INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET]):
    raise RuntimeError("Missing InfluxDB envs. Fix .env: URL/TOKEN/ORG/BUCKET")

# --- Clients ---
influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# -------------------- helpers --------------------
def _cow_id_from_topic(topic: str) -> str:
    parts = topic.split("/")
    if len(parts) >= 4 and parts[0] == "farm" and parts[1] == "cow":
        return parts[2]
    return "unknown"

def _make_mqtt_client():
    try:
        return mqtt.Client(
            client_id="influxdb_consumer",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
    except Exception:
        return mqtt.Client(client_id="influxdb_consumer")

def _parse_sensor_ts(data: dict) -> datetime | None:
    """
    Accepts:
      - ts / t_sensor as ISO8601 string
      - ts_ms / t_sensor_ms as epoch milliseconds
      - ts_s / t_sensor_s as epoch seconds
    Returns datetime in UTC or None if missing/unparseable.
    """
    for key in ("t_sensor", "ts"):
        if key in data:
            v = data[key]
            # ISO string
            if isinstance(v, str):
                try:
                    # allow "Z"
                    s = v.replace("Z", "+00:00")
                    return datetime.fromisoformat(s).astimezone(timezone.utc)
                except Exception:
                    return None
            # numeric seconds
            if isinstance(v, (int, float)):
                # heuristic: big => ms
                try:
                    if v > 1e12:
                        return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
                    if v > 1e9:
                        return datetime.fromtimestamp(v, tz=timezone.utc)
                except Exception:
                    return None

    for key in ("t_sensor_ms", "ts_ms"):
        if key in data and isinstance(data[key], (int, float)):
            try:
                return datetime.fromtimestamp(float(data[key]) / 1000.0, tz=timezone.utc)
            except Exception:
                return None

    for key in ("t_sensor_s", "ts_s"):
        if key in data and isinstance(data[key], (int, float)):
            try:
                return datetime.fromtimestamp(float(data[key]), tz=timezone.utc)
            except Exception:
                return None

    return None

def _get_temperature(data: dict) -> float | None:
    """
    Accept common keys:
      temp, temperature, t
    """
    for k in ("temperature", "temp", "t"):
        if k in data:
            try:
                return float(data[k])
            except Exception:
                return None
    return None

# -------------------- in-memory state --------------------
# Missingness / gaps
state_last_seen = {}  # cow_id -> datetime (ingest time)
state_expected = defaultdict(int)  # cow_id -> expected msg count (approx)
state_received = defaultdict(int)  # cow_id -> received msg count
state_missing  = defaultdict(int)  # cow_id -> missing msg count (approx)
state_gap_count = defaultdict(int) # cow_id -> number of large gaps

# Deviation score for temperature (EWMA mean & variance)
# We track mean and variance via EWMA updates: var <- (1-a)*(var + a*(x-mean_prev)^2)
temp_ewma_mean = {}
temp_ewma_var  = {}

# simple message counter to throttle metrics writes
msg_counter = defaultdict(int)

def _write_points(points):
    if not points:
        return
    write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=points)

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(MQTT_SENSORS_TOPIC)
        client.subscribe(MQTT_PRED_TOPIC)
        print(f"Subscribed to: {MQTT_SENSORS_TOPIC}")
        print(f"Subscribed to: {MQTT_PRED_TOPIC}")
    else:
        print(f"Failed to connect, reason code {reason_code}")

def on_message(client, userdata, msg):
    payload_txt = msg.payload.decode("utf-8", errors="replace")
    try:
        data = json.loads(payload_txt)
    except json.JSONDecodeError:
        print(f"[warn] bad JSON on {msg.topic}: {payload_txt}")
        return

    cow_id = _cow_id_from_topic(msg.topic)
    now = datetime.now(timezone.utc)

    try:
        # -------- sensors --------
        if msg.topic.endswith("/sensors"):
            # required accel
            if not all(k in data for k in ("ax", "ay", "az")):
                return

            ax = float(data["ax"])
            ay = float(data["ay"])
            az = float(data["az"])

            # optional temperature
            temp = _get_temperature(data)  # may be None

            # latency (only if sensor timestamp exists)
            t_sensor = _parse_sensor_ts(data)  # may be None
            ingest_latency_ms = None
            if t_sensor is not None:
                ingest_latency_ms = max(0.0, (now - t_sensor).total_seconds() * 1000.0)

            # ---- missingness/gap accounting (approx, light) ----
            # Count as received
            state_received[cow_id] += 1

            if cow_id in state_last_seen:
                gap_s = (now - state_last_seen[cow_id]).total_seconds()
                # expected intervals passed
                if gap_s > GAP_FACTOR * EXPECTED_INTERVAL_S:
                    # approximate how many messages could be missing in that gap
                    expected_in_gap = int(gap_s / EXPECTED_INTERVAL_S)
                    missing_in_gap = max(0, expected_in_gap - 1)  # minus the one we just got
                    if missing_in_gap > 0:
                        state_missing[cow_id] += missing_in_gap
                        state_gap_count[cow_id] += 1
                # also track expected count roughly
                state_expected[cow_id] += max(1, int(gap_s / EXPECTED_INTERVAL_S))
            else:
                state_expected[cow_id] += 1  # first observation

            state_last_seen[cow_id] = now
            msg_counter[cow_id] += 1

            # ---- write sensor point ----
            p_sensor = (
                Point(MEAS_SENSORS)
                .tag("cow_id", cow_id)
                .field("ax", ax).field("ay", ay).field("az", az)
                .time(now, WritePrecision.NS)
            )
            if temp is not None:
                p_sensor = p_sensor.field("temperature", temp)

            # ---- deviation score (only if temp exists) ----
            # EWMA mean/var per cow
            dev_score = None
            if temp is not None:
                if cow_id not in temp_ewma_mean:
                    temp_ewma_mean[cow_id] = temp
                    temp_ewma_var[cow_id] = 0.0
                else:
                    m_prev = temp_ewma_mean[cow_id]
                    # update mean
                    m_new = (1.0 - DEV_ALPHA) * m_prev + DEV_ALPHA * temp
                    # update variance (EWMA of squared deviation)
                    v_prev = temp_ewma_var[cow_id]
                    v_new = (1.0 - DEV_ALPHA) * (v_prev + DEV_ALPHA * (temp - m_prev) ** 2)
                    temp_ewma_mean[cow_id] = m_new
                    temp_ewma_var[cow_id] = max(0.0, v_new)

                std = math.sqrt(max(temp_ewma_var[cow_id], DEV_MIN_STD ** 2))
                z = (temp - temp_ewma_mean[cow_id]) / std
                # clip to avoid crazy spikes
                z = max(-DEV_Z_CLIP, min(DEV_Z_CLIP, z))
                dev_score = float(abs(z))
                p_sensor = p_sensor.field("deviation_score", dev_score)

            # write sensors
            points = [p_sensor]

            # ---- write pipeline metrics every N messages (cheap) ----
            if msg_counter[cow_id] % METRICS_EVERY_N_MESSAGES == 0:
                expected = state_expected[cow_id]
                received = state_received[cow_id]
                missing = state_missing[cow_id]
                missingness_rate = None
                if expected > 0:
                    # missingness = missing / (missing + received) also ok
                    # here use missing / (expected) in same scale
                    missingness_rate = float(missing / max(1, expected))

                p_metrics = (
                    Point(MEAS_METRICS)
                    .tag("cow_id", cow_id)
                    .field("expected_count", int(expected))
                    .field("received_count", int(received))
                    .field("missing_count", int(missing))
                    .field("gap_count", int(state_gap_count[cow_id]))
                    .time(now, WritePrecision.NS)
                )
                if missingness_rate is not None:
                    p_metrics = p_metrics.field("missingness_rate", missingness_rate)
                if ingest_latency_ms is not None:
                    p_metrics = p_metrics.field("ingest_latency_ms", float(ingest_latency_ms))
                points.append(p_metrics)

            _write_points(points)

            # log
            msg = f"→ wrote SENSOR {cow_id}: ax={ax:.3f}, ay={ay:.3f}, az={az:.3f}"
            if temp is not None:
                msg += f", temp={temp:.2f}"
            if dev_score is not None:
                msg += f", dev={dev_score:.3f}"
            if ingest_latency_ms is not None:
                msg += f", lat={ingest_latency_ms:.1f}ms"
            print(msg)
            return

        # -------- predictions --------
        if msg.topic.endswith("/predictions"):
            label = str(data.get("label", "unknown"))
            conf  = float(data.get("confidence", 0.0))
            model = str(data.get("model", "unknown"))

            p_pred = (
                Point(MEAS_PRED)
                .tag("cow_id", cow_id)
                .tag("model", model)
                .field("label", label)
                .field("confidence", conf)
                .time(now, WritePrecision.NS)
            )

            if data.get("label_id") is not None:
                try:
                    p_pred = p_pred.field("label_id", int(data["label_id"]))
                except Exception:
                    pass

            _write_points([p_pred])
            print(f"→ wrote PRED  {cow_id}: model={model}, label={label}, conf={conf:.3f}")
            return

    except Exception as e:
        print("[error]", e)

def main():
    client = _make_mqtt_client()
    client.on_connect = on_connect
    client.on_message = on_message
    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT} ...")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_forever()

if __name__ == "__main__":
    main()

