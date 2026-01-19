# src/predict/predictor_rf.py
# Real-time RF predictor (Milestone 2)
# - Subscribes to farm/cow/cow_01/sensors (JSON with ax/ay/az)
# - Uses 50-sample window (~5s @ 10 Hz)
# - Extracts 15 stats features: mean/std/min/max/var for ax/ay/az
# - Loads models/baseline_rf.pkl (+ models/label_map.json if present)
# - Publishes prediction to farm/cow/cow_01/predictions:
#   {"model":"rf","label":"...","confidence":0.0~1.0, "label_id":optional_int}

import os
import json
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import paho.mqtt.client as mqtt

# ---------- Defaults (env-overridable) ----------
BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT   = int(os.getenv("MQTT_PORT", "1883"))

SENSORS_TOPIC = os.getenv("MQTT_TOPIC", "farm/cow/cow_01/sensors")
PRED_TOPIC    = os.getenv("MQTT_PRED_TOPIC", "farm/cow/cow_01/predictions")

WINDOW_SIZE = int(os.getenv("RF_WINDOW", "50"))

# Throttle: predict/publish once per N sensor messages (10 @ 10Hz ≈ 1 prediction/sec)
PRED_EVERY = int(os.getenv("RF_PRED_EVERY", "10"))

# Resolve paths robustly: src/predict/predictor_rf.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("RF_MODEL", str(PROJECT_ROOT / "models" / "baseline_rf.pkl")))
LABELS_JSON = Path(os.getenv("RF_LABELS", str(PROJECT_ROOT / "models" / "label_map.json")))
LABEL_ENCODER = Path(os.getenv("RF_LABEL_ENCODER", str(PROJECT_ROOT / "results" / "checkpoints" / "label_encoder.joblib")))

FEATURE_ORDER = [
    "ax_mean","ax_std","ax_min","ax_max","ax_var",
    "ay_mean","ay_std","ay_min","ay_max","ay_var",
    "az_mean","az_std","az_min","az_max","az_var",
]

# ---------- Feature extraction ----------
def make_features(win_np: np.ndarray) -> pd.DataFrame:
    if win_np.ndim != 2 or win_np.shape[1] != 3:
        raise ValueError(f"Window must be (T,3). Got {win_np.shape}")

    ax = win_np[:, 0].astype(float)
    ay = win_np[:, 1].astype(float)
    az = win_np[:, 2].astype(float)

    def stats(v: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(v)),
            "std":  float(np.std(v, ddof=0)),
            "min":  float(np.min(v)),
            "max":  float(np.max(v)),
            "var":  float(np.var(v, ddof=0)),
        }

    sx, sy, sz = stats(ax), stats(ay), stats(az)
    row = {
        "ax_mean": sx["mean"], "ax_std": sx["std"], "ax_min": sx["min"], "ax_max": sx["max"], "ax_var": sx["var"],
        "ay_mean": sy["mean"], "ay_std": sy["std"], "ay_min": sy["min"], "ay_max": sy["max"], "ay_var": sy["var"],
        "az_mean": sz["mean"], "az_std": sz["std"], "az_min": sz["min"], "az_max": sz["max"], "az_var": sz["var"],
    }
    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

def _load_model():
    if not MODEL_PATH.exists():
        raise SystemExit(f"[FATAL] RF model not found: {MODEL_PATH}")
    m = joblib.load(MODEL_PATH)
    if isinstance(m, dict):
        for k in ("model", "clf", "classifier"):
            if k in m:
                return m[k]
    return m

def _load_classes():
    # 1) models/label_map.json with {"classes":[...]}
    if LABELS_JSON.exists():
        try:
            obj = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
            classes = obj.get("classes")
            if isinstance(classes, list) and len(classes) > 0:
                return [str(c) for c in classes]
        except Exception:
            pass

    # 2) LabelEncoder joblib
    if LABEL_ENCODER.exists():
        try:
            le = joblib.load(LABEL_ENCODER)
            if hasattr(le, "classes_"):
                return [str(c) for c in list(le.classes_)]
        except Exception:
            pass

    return None

def _to_label(pred, classes):
    try:
        if isinstance(pred, np.generic):
            pred = pred.item()
        if classes is not None and isinstance(pred, int):
            idx = int(pred)
            if 0 <= idx < len(classes):
                return str(classes[idx])
        return str(pred)
    except Exception:
        return str(pred)

def _make_mqtt_client():
    # tolerant across paho-mqtt versions
    try:
        return mqtt.Client(
            client_id="prediction_rf",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
    except Exception:
        return mqtt.Client(client_id="prediction_rf")

def main():
    parser = argparse.ArgumentParser(description="RF real-time predictor (Milestone 2)")
    parser.add_argument("--broker", default=BROKER)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--sensors-topic", default=SENSORS_TOPIC)
    parser.add_argument("--pred-topic", default=PRED_TOPIC)
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--pred-every", type=int, default=PRED_EVERY)
    args = parser.parse_args()

    model = _load_model()
    classes = _load_classes()
    ring = deque(maxlen=args.window)

    msg_count = 0  # for throttling

    print("[ok] RF predictor ready")
    print(f"  broker     : {args.broker}:{args.port}")
    print(f"  sensors    : {args.sensors_topic}")
    print(f"  predictions: {args.pred_topic}")
    print(f"  model      : {MODEL_PATH}")
    print(f"  window     : {args.window} samples")
    print(f"  pred_every : every {args.pred_every} sensor msgs")
    print(f"  classes    : {classes}")

    client = _make_mqtt_client()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print("[ok] connected")
            client.subscribe(args.sensors_topic)
            print(f"[ok] subscribed: {args.sensors_topic}")
        else:
            print(f"[err] connect failed rc={reason_code}")

    def on_message(client, userdata, msg):
        nonlocal msg_count

        # parse JSON safely
        try:
            payload = json.loads(msg.payload.decode("utf-8", errors="ignore"))
        except Exception:
            return

        # accept only sensor payloads
        if not all(k in payload for k in ("ax", "ay", "az")):
            return

        try:
            ax = float(payload["ax"])
            ay = float(payload["ay"])
            az = float(payload["az"])
        except Exception:
            return

        ring.append([ax, ay, az])
        if len(ring) < args.window:
            return

        msg_count += 1
        if args.pred_every > 1 and (msg_count % args.pred_every != 0):
            return  # throttle

        try:
            win = np.array(ring, dtype=np.float32)
            X = make_features(win)

            y_pred = model.predict(X)[0]
            label = _to_label(y_pred, classes)

            conf = 0.0
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                conf = float(np.max(proba))

            out = {
                "model": "rf",
                "label": label,
                "confidence": conf,
            }
            if isinstance(y_pred, (int, np.integer)):
                out["label_id"] = int(y_pred)

            client.publish(args.pred_topic, json.dumps(out), qos=0, retain=False)
            print(f"Current predicted activity: {label} (conf={conf:.3f})")

        except Exception as e:
            print("[err] predict:", e)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(args.broker, args.port, keepalive=60)
    client.loop_forever()

if __name__ == "__main__":
    main()

