#!/usr/bin/env python3
import os, json, ssl, time, base64, threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import paho.mqtt.client as mqtt
from dotenv import load_dotenv, find_dotenv

# ---------- .env loading ----------
ENV_PATH = find_dotenv(usecwd=True)
if not ENV_PATH:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parents[1]
    ENV_PATH = str(PROJECT_ROOT / ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ---------- TTS MQTT ----------
TTS_APP_ID   = os.getenv("TTS_APP_ID")
TTS_TENANT   = os.getenv("TTS_TENANT", "ttn")
TTS_REGION   = os.getenv("TTS_REGION", "eu1")
TTS_HOST     = os.getenv("TTS_HOST", f"{TTS_REGION}.cloud.thethings.network")
TTS_PORT     = int(os.getenv("TTS_PORT", "8883"))
TTS_TLS      = os.getenv("TTS_TLS", "1") == "1"

# Username usually: "<app-id>@<tenant>"  (or your industries tenant)
TTS_USERNAME = os.getenv("TTS_USERNAME") or (f"{TTS_APP_ID}@{TTS_TENANT}" if TTS_APP_ID else None)

# MQTT password is API key
TTS_API_KEY  = os.getenv("TTS_API_KEY") or os.getenv("TTS_PASSWORD")

# uplink topic: v3/<app-id>@<tenant>/devices/+/up
TTS_UPLINK_TOPIC = os.getenv("TTS_UPLINK_TOPIC") or (f"v3/{TTS_USERNAME}/devices/+/up" if TTS_USERNAME else None)

# ---------- Local MQTT ----------
MQTT_BROKER  = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT    = int(os.getenv("MQTT_PORT", "1883"))

LOCAL_SENSORS_TMPL = os.getenv("LOCAL_SENSORS_TMPL", "farm/cow/{cow_id}/sensors")
LOCAL_RAW_TMPL     = os.getenv("LOCAL_RAW_TMPL",     "farm/cow/{cow_id}/uplink_raw")

# Debug flag (accept BOTH names)
DEBUG_PRINT = (os.getenv("BRIDGE_DEBUG", "") == "1") or (os.getenv("DEBUG_PRINT", "") == "1")

# Heartbeat / reconnect / QoS
HEARTBEAT_SEC = int(os.getenv("BRIDGE_HEARTBEAT_SEC", "10"))
LOCAL_QOS     = int(os.getenv("BRIDGE_LOCAL_QOS", "1"))   # <-- default QoS=1 (more reliable)
TTS_SUB_QOS   = int(os.getenv("BRIDGE_TTS_QOS", "0"))     # TTS uplink often OK with 0
KEEPALIVE     = int(os.getenv("BRIDGE_KEEPALIVE", "60"))

# If you want: wait until publish done (QoS1/2 only). Usually not needed.
WAIT_PUBLISH  = os.getenv("BRIDGE_WAIT_PUBLISH", "0") == "1"


def require_env():
    missing = []
    if not TTS_USERNAME: missing.append("TTS_USERNAME (or TTS_APP_ID + TTS_TENANT)")
    if not TTS_API_KEY: missing.append("TTS_API_KEY")
    if not TTS_UPLINK_TOPIC: missing.append("TTS_UPLINK_TOPIC")
    if missing:
        raise RuntimeError("Missing envs: " + ", ".join(missing))


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def cow_id_from_uplink(msg_json: Dict[str, Any]) -> str:
    return (
        msg_json.get("end_device_ids", {}).get("device_id")
        or msg_json.get("end_device_ids", {}).get("dev_eui")
        or "unknown"
    )


def extract_decoded_payload(msg_json: Dict[str, Any]) -> Dict[str, Any]:
    upl = msg_json.get("uplink_message", {}) or {}
    dp = upl.get("decoded_payload")
    return dp if isinstance(dp, dict) else {}


def extract_raw_uplink(msg_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Always available even if no payload formatter is set (frm_payload is base64).
    """
    upl = msg_json.get("uplink_message", {}) or {}
    frm = upl.get("frm_payload")  # base64 string
    f_port = upl.get("f_port")
    rx = upl.get("rx_metadata", [])
    received_at = msg_json.get("received_at")

    out = {
        "ts": iso_now(),
        "received_at": received_at,
        "f_port": f_port,
        "frm_payload_b64": frm,
    }

    if isinstance(frm, str):
        try:
            raw_bytes = base64.b64decode(frm)
            out["frm_payload_hex"] = raw_bytes.hex()
        except Exception:
            pass

    if isinstance(rx, list) and len(rx) > 0 and isinstance(rx[0], dict):
        gw = rx[0].get("gateway_ids", {}) or {}
        out["gateway_id"] = gw.get("gateway_id")
        out["rssi"] = rx[0].get("rssi")
        out["snr"]  = rx[0].get("snr")

    return out


def normalize_sensors(decoded: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map into your expected schema: ax/ay/az (+ optional temp, ts)
    """
    if all(k in decoded for k in ("ax", "ay", "az")):
        out = {"ax": decoded["ax"], "ay": decoded["ay"], "az": decoded["az"]}
        if "temp" in decoded: out["temp"] = decoded["temp"]
        if "temperature" in decoded: out["temp"] = decoded["temperature"]
        if "ts" in decoded: out["ts"] = decoded["ts"]
        return out

    if isinstance(decoded.get("accel"), dict):
        a = decoded["accel"]
        if all(k in a for k in ("x", "y", "z")):
            out = {"ax": a["x"], "ay": a["y"], "az": a["z"]}
            if "temp" in decoded: out["temp"] = decoded["temp"]
            if "temperature" in decoded: out["temp"] = decoded["temperature"]
            return out

    if isinstance(decoded.get("acc"), dict):
        a = decoded["acc"]
        if all(k in a for k in ("ax", "ay", "az")):
            out = {"ax": a["ax"], "ay": a["ay"], "az": a["az"]}
            if "temp" in decoded: out["temp"] = decoded["temp"]
            if "temperature" in decoded: out["temp"] = decoded["temperature"]
            return out

    return None


def _make_client(client_id: str):
    # tolerate paho-mqtt v1/v2 differences
    try:
        return mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except Exception:
        return mqtt.Client(client_id=client_id)


def make_local_client():
    c = _make_client("tts_bridge_local")
    # auto reconnect delay
    try:
        c.reconnect_delay_set(min_delay=1, max_delay=30)
    except Exception:
        pass
    return c


def make_tts_client():
    c = _make_client("tts_bridge_tts")
    c.username_pw_set(TTS_USERNAME, TTS_API_KEY)

    if TTS_TLS:
        ctx = ssl.create_default_context()
        c.tls_set_context(ctx)

    try:
        c.reconnect_delay_set(min_delay=1, max_delay=60)
    except Exception:
        pass

    if DEBUG_PRINT:
        def _on_log(client, userdata, level, buf):
            print("[tts][log]", buf)
        c.on_log = _on_log

    return c


def main():
    require_env()

    print("\n[tts_bridge config]")
    print(" TTS_HOST         =", f"{TTS_HOST}:{TTS_PORT} (TLS={TTS_TLS})")
    print(" TTS_USERNAME      =", TTS_USERNAME)
    print(" TTS_UPLINK_TOPIC  =", TTS_UPLINK_TOPIC, f"(qos={TTS_SUB_QOS})")
    print(" LOCAL MQTT        =", f"{MQTT_BROKER}:{MQTT_PORT}", f"(pub qos={LOCAL_QOS})")
    print(" LOCAL SENSORS TMPL=", LOCAL_SENSORS_TMPL)
    print(" LOCAL RAW TMPL    =", LOCAL_RAW_TMPL)
    print(" DEBUG_PRINT       =", DEBUG_PRINT)
    print(" HEARTBEAT_SEC     =", HEARTBEAT_SEC)
    print(" WAIT_PUBLISH      =", WAIT_PUBLISH, "\n")

    local = make_local_client()
    tts = make_tts_client()

    # simple runtime state + stats (helps you prove “bridge is alive”)
    state = {
        "last_uplink_ts": 0.0,
        "uplink_total": 0,
        "decoded_ok": 0,
        "decoded_empty": 0,
        "local_pub_ok": 0,
        "local_pub_fail": 0,
    }

    # ---- local callbacks ----
    def on_local_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print("[local] connected")
        else:
            print("[local] connect failed:", reason_code)

    def on_local_disconnect(client, userdata, reason_code, properties=None):
        print("[local] disconnected rc=", reason_code)

    local.on_connect = on_local_connect
    local.on_disconnect = on_local_disconnect

    # connect local first
    local.connect(MQTT_BROKER, MQTT_PORT, keepalive=KEEPALIVE)
    local.loop_start()

    # ---- tts callbacks ----
    def on_tts_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print("[tts] connected")
            client.subscribe(TTS_UPLINK_TOPIC, qos=TTS_SUB_QOS)
            print("[tts] subscribed:", TTS_UPLINK_TOPIC)
        else:
            print("[tts] connect failed:", reason_code)

    def on_tts_disconnect(client, userdata, reason_code, properties=None):
        print("[tts] disconnected rc=", reason_code)

    def _publish_local(topic: str, payload: str) -> bool:
        info = local.publish(topic, payload, qos=LOCAL_QOS, retain=False)
        if info.rc != 0:
            state["local_pub_fail"] += 1
            return False
        if WAIT_PUBLISH and LOCAL_QOS > 0:
            try:
                info.wait_for_publish(timeout=5)
            except Exception:
                pass
        state["local_pub_ok"] += 1
        return True

    def on_tts_message(client, userdata, msg):
        state["last_uplink_ts"] = time.time()
        state["uplink_total"] += 1

        try:
            raw = msg.payload.decode("utf-8", errors="replace")
            j = json.loads(raw)
        except Exception as e:
            if DEBUG_PRINT:
                print("[tts] bad json:", e)
            return

        cow_id = cow_id_from_uplink(j)

        decoded = extract_decoded_payload(j)
        sensors = normalize_sensors(decoded)

        # If no decoded payload: still publish RAW for debugging/visibility
        if sensors is None:
            state["decoded_empty"] += 1
            raw_out = extract_raw_uplink(j)
            topic_raw = LOCAL_RAW_TMPL.format(cow_id=cow_id)
            ok = _publish_local(topic_raw, json.dumps(raw_out, ensure_ascii=False))
            if DEBUG_PRINT:
                print(f"[bridge][raw] ok={ok} {cow_id} -> {topic_raw}")
            return

        state["decoded_ok"] += 1
        sensors.setdefault("ts", iso_now())

        topic = LOCAL_SENSORS_TMPL.format(cow_id=cow_id)
        payload_out = json.dumps(sensors, ensure_ascii=False)

        ok = _publish_local(topic, payload_out)
        if DEBUG_PRINT:
            print(f"[bridge] ok={ok} {cow_id} -> {topic} : {payload_out}")

    tts.on_connect = on_tts_connect
    tts.on_disconnect = on_tts_disconnect
    tts.on_message = on_tts_message

    # ---- heartbeat ----
    def heartbeat():
        while True:
            time.sleep(HEARTBEAT_SEC)
            if state["last_uplink_ts"] <= 0:
                print("[hb] running… (no uplink received yet)")
            else:
                age = time.time() - state["last_uplink_ts"]
                print(
                    "[hb] running… "
                    f"last_uplink={age:.1f}s "
                    f"total={state['uplink_total']} "
                    f"decoded_ok={state['decoded_ok']} "
                    f"decoded_empty={state['decoded_empty']} "
                    f"pub_ok={state['local_pub_ok']} "
                    f"pub_fail={state['local_pub_fail']}"
                )

    threading.Thread(target=heartbeat, daemon=True).start()

    # ---- connect loop ----
    while True:
        try:
            print(f"[tts] connecting to {TTS_HOST}:{TTS_PORT} ...")
            tts.connect(TTS_HOST, TTS_PORT, keepalive=KEEPALIVE)
            tts.loop_forever()
        except KeyboardInterrupt:
            print("\n[bye] stopping bridge...")
            break
        except Exception as e:
            print("[tts] error:", e)
            time.sleep(3)

    try:
        local.loop_stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()

