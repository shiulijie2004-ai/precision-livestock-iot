#!/usr/bin/env python3
"""
tts_bridge.py

Purpose:
  Subscribe to The Things Stack (TTS) MQTT uplinks, extract decoded_payload,
  normalize your Heltec payload format, then publish clean sensor JSON to local MQTT.

Input from TTS decoded_payload can be like:
{
  "ax_g": -0.056,
  "ay_g": 0.015,
  "az_g": -0.994,
  "counter": 0,
  "ds18b20_ok": true,
  "ds18b20_temp_c": 30.43,
  "gx_dps": 0.29,
  "gy_dps": -1.04,
  "gz_dps": -0.26,
  "mpu_ok": true,
  "mpu_temp_c": 32.15
}

Output to local MQTT topic farm/cow/<device_id>/sensors:
{
  "ts": "...",
  "counter": 0,
  "mpu_ok": true,
  "ds18b20_ok": true,
  "ax": -0.056,
  "ay": 0.015,
  "az": -0.994,
  "ax_g": -0.056,
  "ay_g": 0.015,
  "az_g": -0.994,
  "gx": 0.29,
  "gy": -1.04,
  "gz": -0.26,
  "gx_dps": 0.29,
  "gy_dps": -1.04,
  "gz_dps": -0.26,
  "temperature": 30.43,
  "temp": 30.43,
  "ds18b20_temp_c": 30.43,
  "mpu_temp_c": 32.15,
  "temperature_source": "ds18b20"
}
"""

import base64
import json
import os
import ssl
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt
from dotenv import find_dotenv, load_dotenv


# ---------- .env loading ----------
def load_environment() -> str:
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        script_dir = Path(__file__).resolve().parent
        # If file is under project/src/..., parents[1] is project root.
        # If file is directly in project root, this fallback still stays safe.
        try:
            project_root = script_dir.parents[1]
        except IndexError:
            project_root = script_dir
        env_path = str(project_root / ".env")
    load_dotenv(dotenv_path=env_path, override=True)
    return env_path


ENV_PATH = load_environment()


# ---------- TTS MQTT ----------
TTS_APP_ID = os.getenv("TTS_APP_ID")
TTS_TENANT = os.getenv("TTS_TENANT", "ttn")
TTS_REGION = os.getenv("TTS_REGION", "eu1")
TTS_HOST = os.getenv("TTS_HOST", f"{TTS_REGION}.cloud.thethings.network")
TTS_PORT = int(os.getenv("TTS_PORT", "8883"))
TTS_TLS = os.getenv("TTS_TLS", "1") == "1"

# TTS username is normally: <application-id>@<tenant>
# Example public TTS: my-app@ttn
# Example TTI Cloud tenant: my-app@my-tenant
TTS_USERNAME = os.getenv("TTS_USERNAME") or (f"{TTS_APP_ID}@{TTS_TENANT}" if TTS_APP_ID else None)
TTS_API_KEY = os.getenv("TTS_API_KEY") or os.getenv("TTS_PASSWORD")
TTS_UPLINK_TOPIC = os.getenv("TTS_UPLINK_TOPIC") or (
    f"v3/{TTS_USERNAME}/devices/+/up" if TTS_USERNAME else None
)


# ---------- Local MQTT ----------
MQTT_BROKER = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
LOCAL_SENSORS_TMPL = os.getenv("LOCAL_SENSORS_TMPL", "farm/cow/{cow_id}/sensors")
LOCAL_RAW_TMPL = os.getenv("LOCAL_RAW_TMPL", "farm/cow/{cow_id}/uplink_raw")

DEBUG_PRINT = (os.getenv("BRIDGE_DEBUG", "") == "1") or (os.getenv("DEBUG_PRINT", "") == "1")
HEARTBEAT_SEC = int(os.getenv("BRIDGE_HEARTBEAT_SEC", "10"))
LOCAL_QOS = int(os.getenv("BRIDGE_LOCAL_QOS", "1"))
TTS_SUB_QOS = int(os.getenv("BRIDGE_TTS_QOS", "0"))
KEEPALIVE = int(os.getenv("BRIDGE_KEEPALIVE", "60"))
WAIT_PUBLISH = os.getenv("BRIDGE_WAIT_PUBLISH", "0") == "1"


# ---------- helpers ----------
def require_env() -> None:
    missing = []
    if not TTS_USERNAME:
        missing.append("TTS_USERNAME or TTS_APP_ID + TTS_TENANT")
    if not TTS_API_KEY:
        missing.append("TTS_API_KEY")
    if not TTS_UPLINK_TOPIC:
        missing.append("TTS_UPLINK_TOPIC")
    if missing:
        raise RuntimeError("Missing env variable(s): " + ", ".join(missing))


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_success_rc(reason_code: Any) -> bool:
    """Works with paho-mqtt v1 int return codes and v2 ReasonCode objects."""
    try:
        return int(reason_code) == 0
    except Exception:
        return str(reason_code).lower() in ("success", "normal disconnection", "0")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def cow_id_from_uplink(msg_json: Dict[str, Any]) -> str:
    ids = msg_json.get("end_device_ids", {}) or {}
    return str(ids.get("device_id") or ids.get("dev_eui") or "unknown")


def extract_decoded_payload(msg_json: Dict[str, Any]) -> Dict[str, Any]:
    uplink = msg_json.get("uplink_message", {}) or {}
    decoded = uplink.get("decoded_payload")
    return decoded if isinstance(decoded, dict) else {}


def extract_raw_uplink(msg_json: Dict[str, Any]) -> Dict[str, Any]:
    uplink = msg_json.get("uplink_message", {}) or {}
    frm_payload = uplink.get("frm_payload")
    rx_metadata = uplink.get("rx_metadata", [])

    out: Dict[str, Any] = {
        "ts": iso_now(),
        "received_at": msg_json.get("received_at"),
        "f_port": uplink.get("f_port"),
        "frm_payload_b64": frm_payload,
    }

    if isinstance(frm_payload, str):
        try:
            raw_bytes = base64.b64decode(frm_payload)
            out["frm_payload_hex"] = raw_bytes.hex()
        except Exception as exc:
            out["frm_payload_decode_error"] = str(exc)

    if isinstance(rx_metadata, list) and rx_metadata and isinstance(rx_metadata[0], dict):
        first_rx = rx_metadata[0]
        gateway_ids = first_rx.get("gateway_ids", {}) or {}
        out["gateway_id"] = gateway_ids.get("gateway_id")
        out["rssi"] = first_rx.get("rssi")
        out["snr"] = first_rx.get("snr")

    return out


def first_present(data: Dict[str, Any], keys: tuple) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def normalize_sensors(decoded: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert many possible decoded_payload shapes into one stable local schema.

    Accepts your current keys:
      ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps,
      ds18b20_temp_c, mpu_temp_c, mpu_ok, ds18b20_ok, counter

    Also accepts older/simple keys:
      ax, ay, az, gx, gy, gz, temp, temperature
    """
    if not isinstance(decoded, dict) or not decoded:
        return None

    out: Dict[str, Any] = {"ts": iso_now()}

    # Keep identity/status fields if available
    for key in ("counter", "mpu_ok", "ds18b20_ok", "label"):
        if key in decoded:
            out[key] = decoded[key]

    # Flat accel keys: your new format or older format
    ax = safe_float(first_present(decoded, ("ax_g", "ax", "acc_x", "x")))
    ay = safe_float(first_present(decoded, ("ay_g", "ay", "acc_y", "y")))
    az = safe_float(first_present(decoded, ("az_g", "az", "acc_z", "z")))

    # Nested accel alternatives
    if ax is None or ay is None or az is None:
        for parent in ("accel", "acc", "accelerometer"):
            nested = decoded.get(parent)
            if isinstance(nested, dict):
                ax = safe_float(first_present(nested, ("ax_g", "ax", "x")))
                ay = safe_float(first_present(nested, ("ay_g", "ay", "y")))
                az = safe_float(first_present(nested, ("az_g", "az", "z")))
                if ax is not None and ay is not None and az is not None:
                    break

    if ax is None or ay is None or az is None:
        return None

    # Canonical names used by your consumer / ML code
    out["ax"] = ax
    out["ay"] = ay
    out["az"] = az

    # Unit-specific names for Grafana clarity
    out["ax_g"] = ax
    out["ay_g"] = ay
    out["az_g"] = az

    # Gyroscope: optional
    gx = safe_float(first_present(decoded, ("gx_dps", "gx", "gyro_x")))
    gy = safe_float(first_present(decoded, ("gy_dps", "gy", "gyro_y")))
    gz = safe_float(first_present(decoded, ("gz_dps", "gz", "gyro_z")))

    if gx is not None:
        out["gx"] = gx
        out["gx_dps"] = gx
    if gy is not None:
        out["gy"] = gy
        out["gy_dps"] = gy
    if gz is not None:
        out["gz"] = gz
        out["gz_dps"] = gz

    # Temperatures. Prefer DS18B20 as animal/body/external temp.
    ds_temp = safe_float(first_present(decoded, ("ds18b20_temp_c", "ds_temp_c", "body_temp_c")))
    mpu_temp = safe_float(first_present(decoded, ("mpu_temp_c", "imu_temp_c")))
    generic_temp = safe_float(first_present(decoded, ("temperature", "temp", "t")))

    if ds_temp is not None:
        out["temperature"] = ds_temp
        out["temp"] = ds_temp
        out["ds18b20_temp_c"] = ds_temp
        out["temperature_source"] = "ds18b20"
    elif generic_temp is not None:
        out["temperature"] = generic_temp
        out["temp"] = generic_temp
        out["temperature_source"] = "generic"

    if mpu_temp is not None:
        out["mpu_temp_c"] = mpu_temp

    # Preserve original timestamps if payload formatter sends them
    for key in ("t_sensor", "t_sensor_ms", "t_sensor_s", "ts_ms", "ts_s"):
        if key in decoded:
            out[key] = decoded[key]

    # If decoded has its own ISO ts, keep it as t_sensor and retain bridge ts separately.
    if "ts" in decoded:
        out["t_sensor"] = decoded["ts"]

    return out


def make_client(client_id: str) -> mqtt.Client:
    # paho-mqtt v2 supports callback_api_version; v1 does not.
    try:
        return mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except Exception:
        return mqtt.Client(client_id=client_id)


def make_local_client() -> mqtt.Client:
    client = make_client("tts_bridge_local")
    try:
        client.reconnect_delay_set(min_delay=1, max_delay=30)
    except Exception:
        pass
    return client


def make_tts_client() -> mqtt.Client:
    client = make_client("tts_bridge_tts")
    client.username_pw_set(TTS_USERNAME, TTS_API_KEY)

    if TTS_TLS:
        client.tls_set_context(ssl.create_default_context())

    try:
        client.reconnect_delay_set(min_delay=1, max_delay=60)
    except Exception:
        pass

    if DEBUG_PRINT:
        def on_log(client, userdata, level, buf):
            print("[tts][log]", buf)
        client.on_log = on_log

    return client


# ---------- main ----------
def main() -> None:
    require_env()

    print("\n[tts_bridge config]")
    print(" ENV_PATH          =", ENV_PATH)
    print(" TTS_HOST          =", f"{TTS_HOST}:{TTS_PORT} TLS={TTS_TLS}")
    print(" TTS_USERNAME      =", TTS_USERNAME)
    print(" TTS_UPLINK_TOPIC  =", TTS_UPLINK_TOPIC, f"qos={TTS_SUB_QOS}")
    print(" LOCAL MQTT        =", f"{MQTT_BROKER}:{MQTT_PORT}", f"pub_qos={LOCAL_QOS}")
    print(" LOCAL_SENSORS_TMPL=", LOCAL_SENSORS_TMPL)
    print(" LOCAL_RAW_TMPL    =", LOCAL_RAW_TMPL)
    print(" DEBUG_PRINT       =", DEBUG_PRINT)
    print(" HEARTBEAT_SEC     =", HEARTBEAT_SEC)
    print(" WAIT_PUBLISH      =", WAIT_PUBLISH, "\n")

    local = make_local_client()
    tts = make_tts_client()

    state = {
        "last_uplink_ts": 0.0,
        "uplink_total": 0,
        "decoded_ok": 0,
        "decoded_empty": 0,
        "local_pub_ok": 0,
        "local_pub_fail": 0,
    }

    def on_local_connect(client, userdata, flags, reason_code, properties=None):
        if is_success_rc(reason_code):
            print("[local] connected")
        else:
            print("[local] connect failed:", reason_code)

    def on_local_disconnect(client, userdata, reason_code, properties=None):
        print("[local] disconnected rc=", reason_code)

    local.on_connect = on_local_connect
    local.on_disconnect = on_local_disconnect

    def publish_local(topic: str, payload_dict: Dict[str, Any]) -> bool:
        payload = json.dumps(payload_dict, ensure_ascii=False, separators=(",", ":"))
        info = local.publish(topic, payload, qos=LOCAL_QOS, retain=False)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            state["local_pub_fail"] += 1
            print(f"[local][publish failed] rc={info.rc} topic={topic}")
            return False
        if WAIT_PUBLISH and LOCAL_QOS > 0:
            try:
                info.wait_for_publish(timeout=5)
            except TypeError:
                info.wait_for_publish()
            except Exception:
                pass
        state["local_pub_ok"] += 1
        return True

    def on_tts_connect(client, userdata, flags, reason_code, properties=None):
        if is_success_rc(reason_code):
            print("[tts] connected")
            client.subscribe(TTS_UPLINK_TOPIC, qos=TTS_SUB_QOS)
            print("[tts] subscribed:", TTS_UPLINK_TOPIC)
        else:
            print("[tts] connect failed:", reason_code)

    def on_tts_disconnect(client, userdata, reason_code, properties=None):
        print("[tts] disconnected rc=", reason_code)

    def on_tts_message(client, userdata, msg):
        state["last_uplink_ts"] = time.time()
        state["uplink_total"] += 1

        try:
            raw_text = msg.payload.decode("utf-8", errors="replace")
            uplink_json = json.loads(raw_text)
        except Exception as exc:
            print("[tts] bad JSON:", exc)
            return

        cow_id = cow_id_from_uplink(uplink_json)
        decoded = extract_decoded_payload(uplink_json)
        sensors = normalize_sensors(decoded)

        if sensors is None:
            state["decoded_empty"] += 1
            raw_out = extract_raw_uplink(uplink_json)
            raw_out["note"] = "No usable decoded_payload. Check TTS payload formatter."
            topic_raw = LOCAL_RAW_TMPL.format(cow_id=cow_id)
            ok = publish_local(topic_raw, raw_out)
            print(f"[bridge][raw] ok={ok} cow={cow_id} topic={topic_raw}")
            return

        state["decoded_ok"] += 1
        topic = LOCAL_SENSORS_TMPL.format(cow_id=cow_id)
        ok = publish_local(topic, sensors)

        if DEBUG_PRINT:
            print(f"[bridge] ok={ok} cow={cow_id} topic={topic} payload={json.dumps(sensors, ensure_ascii=False)}")
        else:
            temp = sensors.get("temperature")
            gyro_msg = ""
            if all(k in sensors for k in ("gx_dps", "gy_dps", "gz_dps")):
                gyro_msg = f", gyro=({sensors['gx_dps']:.2f},{sensors['gy_dps']:.2f},{sensors['gz_dps']:.2f})"
            temp_msg = f", temp={float(temp):.2f}" if temp is not None else ""
            print(
                f"[bridge] ok={ok} cow={cow_id} "
                f"acc=({sensors['ax']:.3f},{sensors['ay']:.3f},{sensors['az']:.3f})"
                f"{gyro_msg}{temp_msg}"
            )

    tts.on_connect = on_tts_connect
    tts.on_disconnect = on_tts_disconnect
    tts.on_message = on_tts_message

    def heartbeat():
        while True:
            time.sleep(HEARTBEAT_SEC)
            if state["last_uplink_ts"] <= 0:
                print("[hb] running... no TTS uplink received yet")
            else:
                age = time.time() - state["last_uplink_ts"]
                print(
                    "[hb] running... "
                    f"last_uplink={age:.1f}s "
                    f"total={state['uplink_total']} "
                    f"decoded_ok={state['decoded_ok']} "
                    f"decoded_empty={state['decoded_empty']} "
                    f"pub_ok={state['local_pub_ok']} "
                    f"pub_fail={state['local_pub_fail']}"
                )

    try:
        local.connect(MQTT_BROKER, MQTT_PORT, keepalive=KEEPALIVE)
        local.loop_start()
    except Exception as exc:
        raise RuntimeError(f"Cannot connect to local MQTT broker {MQTT_BROKER}:{MQTT_PORT}: {exc}") from exc

    threading.Thread(target=heartbeat, daemon=True).start()

    while True:
        try:
            print(f"[tts] connecting to {TTS_HOST}:{TTS_PORT} ...")
            tts.connect(TTS_HOST, TTS_PORT, keepalive=KEEPALIVE)
            tts.loop_forever(retry_first_connection=True)
        except KeyboardInterrupt:
            print("\n[bye] stopping bridge...")
            break
        except Exception as exc:
            print("[tts] error:", exc)
            time.sleep(3)

    try:
        tts.loop_stop()
    except Exception:
        pass
    try:
        local.loop_stop()
        local.disconnect()
    except Exception:
        pass


if __name__ == "__main__":
    main()
