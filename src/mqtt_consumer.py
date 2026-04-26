#!/usr/bin/env python3
"""
mqtt_consumer.py

Purpose:
  Subscribe to local MQTT sensor messages and write them to InfluxDB.

This fixed version accepts both old/simple keys:
  ax, ay, az, gx, gy, gz, temp, temperature

and your current Heltec/TTS payload keys:
  ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps,
  ds18b20_temp_c, mpu_temp_c, mpu_ok, ds18b20_ok, counter

Recommended local MQTT sensor topic:
  farm/cow/+/sensors
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import paho.mqtt.client as mqtt
from dotenv import find_dotenv, load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


# ---------- .env loading ----------
def load_environment() -> str:
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        script_dir = Path(__file__).resolve().parent
        try:
            project_root = script_dir.parents[1]
        except IndexError:
            project_root = script_dir
        env_path = str(project_root / ".env")
    load_dotenv(dotenv_path=env_path, override=True)
    return env_path


ENV_PATH = load_environment()


# ---------- Config ----------
MQTT_BROKER = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_SENSORS_TOPIC = os.getenv("MQTT_SENSORS_TOPIC", "farm/cow/+/sensors")
MQTT_PRED_TOPIC = os.getenv("MQTT_PRED_TOPIC", "farm/cow/+/predictions")
MQTT_QOS = int(os.getenv("MQTT_CONSUMER_QOS", "1"))

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "animal-data")

MEAS_SENSORS = os.getenv("MEAS_SENSORS", "sensor_reading")
MEAS_PRED = os.getenv("MEAS_PRED", "predictions")
MEAS_METRICS = os.getenv("MEAS_METRICS", "pipeline_metrics")

EXPECTED_INTERVAL_S = float(os.getenv("EXPECTED_PUBLISH_INTERVAL_S", "20.0"))
GAP_FACTOR = float(os.getenv("GAP_FACTOR", "2.0"))
METRICS_EVERY_N_MESSAGES = int(os.getenv("METRICS_EVERY_N_MESSAGES", "10"))

DEV_ALPHA = float(os.getenv("DEV_ALPHA", "0.02"))
DEV_MIN_STD = float(os.getenv("DEV_MIN_STD", "0.05"))
DEV_Z_CLIP = float(os.getenv("DEV_Z_CLIP", "10.0"))

DEBUG_PRINT = os.getenv("CONSUMER_DEBUG", "0") == "1"


# ---------- validation ----------
def require_env() -> None:
    missing = []
    if not INFLUXDB_URL:
        missing.append("INFLUXDB_URL")
    if not INFLUXDB_TOKEN:
        missing.append("INFLUXDB_TOKEN")
    if not INFLUXDB_ORG:
        missing.append("INFLUXDB_ORG")
    if not INFLUXDB_BUCKET:
        missing.append("INFLUXDB_BUCKET")
    if missing:
        raise RuntimeError("Missing InfluxDB env variable(s): " + ", ".join(missing))


require_env()

masked = (INFLUXDB_TOKEN[:6] + "..." + INFLUXDB_TOKEN[-4:]) if INFLUXDB_TOKEN else None
print("\n[consumer config]")
print(" ENV_PATH          =", ENV_PATH)
print(" INFLUXDB_URL      =", INFLUXDB_URL)
print(" INFLUXDB_ORG      =", INFLUXDB_ORG)
print(" INFLUXDB_BUCKET   =", INFLUXDB_BUCKET)
print(" INFLUXDB_TOKEN    =", masked)
print(" MQTT_BROKER       =", f"{MQTT_BROKER}:{MQTT_PORT}")
print(" SUB sensors       =", MQTT_SENSORS_TOPIC, f"qos={MQTT_QOS}")
print(" SUB predictions   =", MQTT_PRED_TOPIC, f"qos={MQTT_QOS}")
print(" MEAS sensors      =", MEAS_SENSORS)
print(" MEAS predictions  =", MEAS_PRED)
print(" MEAS metrics      =", MEAS_METRICS)
print(" EXPECTED_INTERVAL_S =", EXPECTED_INTERVAL_S)
print(" GAP_FACTOR          =", GAP_FACTOR, "\n")


# ---------- clients ----------
influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)


# ---------- helpers ----------
def is_success_rc(reason_code: Any) -> bool:
    try:
        return int(reason_code) == 0
    except Exception:
        return str(reason_code).lower() in ("success", "normal disconnection", "0")


def make_mqtt_client() -> mqtt.Client:
    try:
        client = mqtt.Client(client_id="influxdb_consumer", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except Exception:
        client = mqtt.Client(client_id="influxdb_consumer")
    try:
        client.reconnect_delay_set(min_delay=1, max_delay=30)
    except Exception:
        pass
    return client


def cow_id_from_topic(topic: str) -> str:
    parts = topic.split("/")
    if len(parts) >= 4 and parts[0] == "farm" and parts[1] == "cow":
        return parts[2]
    return "unknown"


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "y", "ok"):
            return True
        if s in ("0", "false", "no", "n", "fail"):
            return False
    return None


def first_present(data: Dict[str, Any], keys: tuple) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def parse_sensor_ts(data: Dict[str, Any]) -> Optional[datetime]:
    """
    Accepts:
      - ts / t_sensor as ISO8601 string
      - ts_ms / t_sensor_ms as epoch milliseconds
      - ts_s / t_sensor_s as epoch seconds
    """
    for key in ("t_sensor", "ts"):
        if key in data:
            value = data[key]
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    return None
            if isinstance(value, (int, float)):
                try:
                    if value > 1e12:
                        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
                    if value > 1e9:
                        return datetime.fromtimestamp(float(value), tz=timezone.utc)
                except Exception:
                    return None

    for key in ("t_sensor_ms", "ts_ms"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
            except Exception:
                return None

    for key in ("t_sensor_s", "ts_s"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except Exception:
                return None

    return None


def normalize_sensor_payload(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return normalized fields or None if accel is missing."""
    ax = safe_float(first_present(data, ("ax", "ax_g", "acc_x", "x")))
    ay = safe_float(first_present(data, ("ay", "ay_g", "acc_y", "y")))
    az = safe_float(first_present(data, ("az", "az_g", "acc_z", "z")))

    if ax is None or ay is None or az is None:
        return None

    gx = safe_float(first_present(data, ("gx", "gx_dps", "gyro_x")))
    gy = safe_float(first_present(data, ("gy", "gy_dps", "gyro_y")))
    gz = safe_float(first_present(data, ("gz", "gz_dps", "gyro_z")))

    # Prefer DS18B20 for actual animal/external temperature.
    ds_temp = safe_float(first_present(data, ("ds18b20_temp_c", "ds_temp_c", "body_temp_c")))
    generic_temp = safe_float(first_present(data, ("temperature", "temp", "t")))
    mpu_temp = safe_float(first_present(data, ("mpu_temp_c", "imu_temp_c")))

    temperature = ds_temp if ds_temp is not None else generic_temp

    out: Dict[str, Any] = {
        "ax": ax,
        "ay": ay,
        "az": az,
        "ax_g": ax,
        "ay_g": ay,
        "az_g": az,
    }

    if gx is not None:
        out["gx"] = gx
        out["gx_dps"] = gx
    if gy is not None:
        out["gy"] = gy
        out["gy_dps"] = gy
    if gz is not None:
        out["gz"] = gz
        out["gz_dps"] = gz

    if temperature is not None:
        out["temperature"] = temperature
        out["temp"] = temperature
        out["temperature_source"] = "ds18b20" if ds_temp is not None else "generic"
    if ds_temp is not None:
        out["ds18b20_temp_c"] = ds_temp
    if mpu_temp is not None:
        out["mpu_temp_c"] = mpu_temp

    counter = safe_int(data.get("counter"))
    if counter is not None:
        out["counter"] = counter

    mpu_ok = safe_bool(data.get("mpu_ok"))
    ds_ok = safe_bool(data.get("ds18b20_ok"))
    if mpu_ok is not None:
        out["mpu_ok"] = mpu_ok
    if ds_ok is not None:
        out["ds18b20_ok"] = ds_ok

    return out


def write_points(points: List[Point]) -> None:
    if points:
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=points)


# ---------- in-memory state ----------
state_last_seen: Dict[str, datetime] = {}
state_expected = defaultdict(int)
state_received = defaultdict(int)
state_missing = defaultdict(int)
state_gap_count = defaultdict(int)
msg_counter = defaultdict(int)

temp_ewma_mean: Dict[str, float] = {}
temp_ewma_var: Dict[str, float] = {}


def update_missingness(cow_id: str, now: datetime) -> None:
    state_received[cow_id] += 1

    if cow_id in state_last_seen:
        gap_s = (now - state_last_seen[cow_id]).total_seconds()
        expected_in_gap = max(1, int(gap_s / EXPECTED_INTERVAL_S)) if EXPECTED_INTERVAL_S > 0 else 1
        state_expected[cow_id] += expected_in_gap

        if EXPECTED_INTERVAL_S > 0 and gap_s > GAP_FACTOR * EXPECTED_INTERVAL_S:
            missing_in_gap = max(0, expected_in_gap - 1)
            if missing_in_gap > 0:
                state_missing[cow_id] += missing_in_gap
                state_gap_count[cow_id] += 1
    else:
        state_expected[cow_id] += 1

    state_last_seen[cow_id] = now
    msg_counter[cow_id] += 1


def update_temperature_deviation(cow_id: str, temperature: Optional[float]) -> Optional[float]:
    if temperature is None:
        return None

    if cow_id not in temp_ewma_mean:
        temp_ewma_mean[cow_id] = temperature
        temp_ewma_var[cow_id] = 0.0
    else:
        previous_mean = temp_ewma_mean[cow_id]
        previous_var = temp_ewma_var[cow_id]
        new_mean = (1.0 - DEV_ALPHA) * previous_mean + DEV_ALPHA * temperature
        new_var = (1.0 - DEV_ALPHA) * (previous_var + DEV_ALPHA * (temperature - previous_mean) ** 2)
        temp_ewma_mean[cow_id] = new_mean
        temp_ewma_var[cow_id] = max(0.0, new_var)

    std = math.sqrt(max(temp_ewma_var[cow_id], DEV_MIN_STD ** 2))
    z_score = (temperature - temp_ewma_mean[cow_id]) / std
    z_score = max(-DEV_Z_CLIP, min(DEV_Z_CLIP, z_score))
    return float(abs(z_score))


# ---------- MQTT callbacks ----------
def on_connect(client, userdata, flags, reason_code, properties=None):
    if is_success_rc(reason_code):
        print("[mqtt] connected")
        client.subscribe(MQTT_SENSORS_TOPIC, qos=MQTT_QOS)
        client.subscribe(MQTT_PRED_TOPIC, qos=MQTT_QOS)
        print(f"[mqtt] subscribed: {MQTT_SENSORS_TOPIC}")
        print(f"[mqtt] subscribed: {MQTT_PRED_TOPIC}")
    else:
        print(f"[mqtt] failed to connect, reason code={reason_code}")


def on_disconnect(client, userdata, reason_code, properties=None):
    print("[mqtt] disconnected rc=", reason_code)


def handle_sensor_message(cow_id: str, data: Dict[str, Any], now: datetime) -> None:
    normalized = normalize_sensor_payload(data)
    if normalized is None:
        print(f"[warn] sensor payload missing accel keys for cow={cow_id}: {data}")
        return

    ax = normalized["ax"]
    ay = normalized["ay"]
    az = normalized["az"]
    temperature = normalized.get("temperature")

    sensor_ts = parse_sensor_ts(data)
    ingest_latency_ms = None
    if sensor_ts is not None:
        ingest_latency_ms = max(0.0, (now - sensor_ts).total_seconds() * 1000.0)

    update_missingness(cow_id, now)
    dev_score = update_temperature_deviation(cow_id, temperature)

    p_sensor = (
        Point(MEAS_SENSORS)
        .tag("cow_id", cow_id)
        .field("ax", float(ax))
        .field("ay", float(ay))
        .field("az", float(az))
        .field("ax_g", float(ax))
        .field("ay_g", float(ay))
        .field("az_g", float(az))
        .time(now, WritePrecision.NS)
    )

    # Optional gyro fields
    for field_name in ("gx", "gy", "gz", "gx_dps", "gy_dps", "gz_dps"):
        value = normalized.get(field_name)
        if value is not None:
            p_sensor = p_sensor.field(field_name, float(value))

    # Optional temperatures
    if temperature is not None:
        p_sensor = p_sensor.field("temperature", float(temperature))
    if normalized.get("ds18b20_temp_c") is not None:
        p_sensor = p_sensor.field("ds18b20_temp_c", float(normalized["ds18b20_temp_c"]))
    if normalized.get("mpu_temp_c") is not None:
        p_sensor = p_sensor.field("mpu_temp_c", float(normalized["mpu_temp_c"]))

    # Optional status/counter fields
    if normalized.get("counter") is not None:
        p_sensor = p_sensor.field("counter", int(normalized["counter"]))
    if normalized.get("mpu_ok") is not None:
        p_sensor = p_sensor.field("mpu_ok", bool(normalized["mpu_ok"]))
    if normalized.get("ds18b20_ok") is not None:
        p_sensor = p_sensor.field("ds18b20_ok", bool(normalized["ds18b20_ok"]))
    if normalized.get("temperature_source") is not None:
        p_sensor = p_sensor.tag("temperature_source", str(normalized["temperature_source"]))

    if dev_score is not None:
        p_sensor = p_sensor.field("deviation_score", float(dev_score))

    points = [p_sensor]

    if msg_counter[cow_id] % METRICS_EVERY_N_MESSAGES == 0:
        expected = state_expected[cow_id]
        received = state_received[cow_id]
        missing = state_missing[cow_id]
        missingness_rate = float(missing / max(1, expected))

        p_metrics = (
            Point(MEAS_METRICS)
            .tag("cow_id", cow_id)
            .field("expected_count", int(expected))
            .field("received_count", int(received))
            .field("missing_count", int(missing))
            .field("gap_count", int(state_gap_count[cow_id]))
            .field("missingness_rate", missingness_rate)
            .time(now, WritePrecision.NS)
        )
        if ingest_latency_ms is not None:
            p_metrics = p_metrics.field("ingest_latency_ms", float(ingest_latency_ms))
        points.append(p_metrics)

    write_points(points)

    log_msg = f"→ wrote SENSOR {cow_id}: ax={ax:.3f}, ay={ay:.3f}, az={az:.3f}"
    if all(k in normalized for k in ("gx_dps", "gy_dps", "gz_dps")):
        log_msg += f", gx={normalized['gx_dps']:.2f}, gy={normalized['gy_dps']:.2f}, gz={normalized['gz_dps']:.2f}"
    if temperature is not None:
        log_msg += f", temp={temperature:.2f}"
    if normalized.get("mpu_temp_c") is not None:
        log_msg += f", mpu_temp={normalized['mpu_temp_c']:.2f}"
    if dev_score is not None:
        log_msg += f", dev={dev_score:.3f}"
    if ingest_latency_ms is not None:
        log_msg += f", latency={ingest_latency_ms:.1f}ms"
    print(log_msg)

    if DEBUG_PRINT:
        print("[debug normalized]", json.dumps(normalized, ensure_ascii=False))


def handle_prediction_message(cow_id: str, data: Dict[str, Any], now: datetime) -> None:
    label = str(data.get("label", "unknown"))
    confidence = safe_float(data.get("confidence"))
    if confidence is None:
        confidence = 0.0
    model = str(data.get("model", "unknown"))

    p_pred = (
        Point(MEAS_PRED)
        .tag("cow_id", cow_id)
        .tag("model", model)
        .field("label", label)
        .field("confidence", float(confidence))
        .time(now, WritePrecision.NS)
    )

    label_id = safe_int(data.get("label_id"))
    if label_id is not None:
        p_pred = p_pred.field("label_id", label_id)

    write_points([p_pred])
    print(f"→ wrote PRED {cow_id}: model={model}, label={label}, conf={confidence:.3f}")


def on_message(client, userdata, msg):
    payload_txt = msg.payload.decode("utf-8", errors="replace")
    try:
        data = json.loads(payload_txt)
    except json.JSONDecodeError:
        print(f"[warn] bad JSON on {msg.topic}: {payload_txt}")
        return

    if not isinstance(data, dict):
        print(f"[warn] JSON payload is not an object on {msg.topic}: {payload_txt}")
        return

    cow_id = cow_id_from_topic(msg.topic)
    now = datetime.now(timezone.utc)

    try:
        if msg.topic.endswith("/sensors"):
            handle_sensor_message(cow_id, data, now)
            return

        if msg.topic.endswith("/predictions"):
            handle_prediction_message(cow_id, data, now)
            return

        print(f"[warn] ignored topic: {msg.topic}")

    except Exception as exc:
        print(f"[error] topic={msg.topic} error={exc}")
        if DEBUG_PRINT:
            raise


def main() -> None:
    client = make_mqtt_client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT} ...")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

    try:
        client.loop_forever(retry_first_connection=True)
    except KeyboardInterrupt:
        print("\n[bye] stopping consumer...")
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
        try:
            influx_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
