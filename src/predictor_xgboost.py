#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor_xgboost.py

Real-time cattle behaviour prediction using exported XGBoost / sklearn joblib model.

Pipeline:
- Subscribe to MQTT sensor topic: farm/cow/+/sensors
- Keep rolling window per cow_id
- Convert latest window into model input features
- Predict behaviour
- Publish prediction to: farm/cow/{cow_id}/predictions

Important:
This version supports BOTH:
1. Flatten/raw-window features:
   Example: window=50, 6 sensor channels
   50 x 6 = 300 features

2. Statistical features fallback:
   Example: 6 channels x 6 stats = 36 features

The script automatically checks model.n_features_in_ and chooses the correct format.
"""

import os
import json
import argparse
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Deque, List, Optional, Any, Tuple

import numpy as np
import paho.mqtt.client as mqtt
import joblib


# ============================================================
# Basic helpers
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_mqtt_client():
    """
    Create a Paho MQTT client compatible with old and new versions.
    """
    try:
        return mqtt.Client(
            client_id="prediction_xgboost",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
    except Exception:
        return mqtt.Client(client_id="prediction_xgboost")


def extract_cow_id_from_topic(topic: str) -> str:
    """
    Example:
    farm/cow/cow1/sensors -> cow1
    """
    parts = [p for p in topic.split("/") if p]

    for i, p in enumerate(parts):
        if p.lower() == "cow" and i + 1 < len(parts):
            return parts[i + 1]

    return parts[2] if len(parts) >= 3 else "unknown"


def safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


# ============================================================
# Sensor extraction
# ============================================================

def get_value_case_insensitive(payload: dict, candidates: List[str]) -> Optional[float]:
    """
    Read payload value using possible key names.

    Supported examples:
    ax_g, ax, acc_x, AccX
    gx_dps, gx, gyro_x, GyroX
    """
    lower_map = {str(k).lower(): k for k in payload.keys()}

    for c in candidates:
        key = c.lower()
        if key in lower_map:
            raw_key = lower_map[key]
            return safe_float(payload.get(raw_key))

    return None


def pick_sensor_vector(payload: dict, feature_keys: List[str]) -> Optional[List[float]]:
    """
    Extract one row of sensor features in exact order.

    Default order:
    ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps
    """

    # Exact user-defined feature keys
    if feature_keys:
        vec = []
        for k in feature_keys:
            v = get_value_case_insensitive(payload, [k])
            if v is None:
                return None
            vec.append(v)
        return vec

    # Flexible default aliases
    ax = get_value_case_insensitive(payload, ["ax_g", "ax", "accx", "acc_x", "AccX"])
    ay = get_value_case_insensitive(payload, ["ay_g", "ay", "accy", "acc_y", "AccY"])
    az = get_value_case_insensitive(payload, ["az_g", "az", "accz", "acc_z", "AccZ"])

    gx = get_value_case_insensitive(payload, ["gx_dps", "gx", "gyrox", "gyro_x", "GyroX"])
    gy = get_value_case_insensitive(payload, ["gy_dps", "gy", "gyroy", "gyro_y", "GyroY"])
    gz = get_value_case_insensitive(payload, ["gz_dps", "gz", "gyroz", "gyro_z", "GyroZ"])

    if ax is None or ay is None or az is None:
        return None

    # Prefer 6 features if gyro exists
    if gx is not None and gy is not None and gz is not None:
        return [ax, ay, az, gx, gy, gz]

    # Fallback accelerometer-only
    return [ax, ay, az]


# ============================================================
# Feature builders
# ============================================================

def build_flatten_features(window_data: np.ndarray) -> np.ndarray:
    """
    Convert rolling window directly into flattened raw features.

    Example:
    window_data shape = [50, 6]

    Output:
    [ax1, ay1, az1, gx1, gy1, gz1,
     ax2, ay2, az2, gx2, gy2, gz2,
     ...
     ax50, ay50, az50, gx50, gy50, gz50]

    Output shape = [1, 300]
    """
    return window_data.astype(np.float32).reshape(1, -1)


def build_stat_features(window_data: np.ndarray) -> np.ndarray:
    """
    Statistical fallback features.

    Per sensor:
    mean, std, min, max, range, energy

    If 6 sensors:
    6 x 6 stats = 36 features
    """
    feats = []

    for i in range(window_data.shape[1]):
        x = window_data[:, i].astype(np.float32)

        mean = float(np.mean(x))
        std = float(np.std(x))
        min_v = float(np.min(x))
        max_v = float(np.max(x))
        range_v = float(max_v - min_v)
        energy = float(np.mean(x ** 2))

        feats.extend([mean, std, min_v, max_v, range_v, energy])

    return np.array(feats, dtype=np.float32).reshape(1, -1)


def get_model_expected_features(model: Any) -> Optional[int]:
    """
    Try to detect number of features expected by sklearn / XGBoost model.
    """
    if hasattr(model, "n_features_in_"):
        try:
            return int(model.n_features_in_)
        except Exception:
            pass

    # Some pipelines store the final estimator
    if hasattr(model, "named_steps"):
        try:
            for step in reversed(list(model.named_steps.values())):
                if hasattr(step, "n_features_in_"):
                    return int(step.n_features_in_)
        except Exception:
            pass

    # Raw XGBoost booster fallback
    try:
        booster = model.get_booster()
        if booster is not None:
            num_features = booster.num_features()
            if num_features:
                return int(num_features)
    except Exception:
        pass

    return None


def build_model_features(
    window_np: np.ndarray,
    expected_features: Optional[int],
    mode: str
) -> Tuple[np.ndarray, str]:
    """
    Build feature row according to mode.

    mode:
    - auto
    - flatten
    - stats
    """

    flat_X = build_flatten_features(window_np)
    stat_X = build_stat_features(window_np)

    if mode == "flatten":
        return flat_X, "flatten"

    if mode == "stats":
        return stat_X, "stats"

    # Auto mode: choose based on model expected feature count
    if expected_features is not None:
        if flat_X.shape[1] == expected_features:
            return flat_X, "flatten"

        if stat_X.shape[1] == expected_features:
            return stat_X, "stats"

        raise ValueError(
            f"Feature shape mismatch. Model expected {expected_features}, "
            f"flatten gives {flat_X.shape[1]}, stats gives {stat_X.shape[1]}. "
            f"Check --window and --feature-keys."
        )

    # If expected unknown, prefer flatten for your current model style
    return flat_X, "flatten"


# ============================================================
# Sidecar metadata
# ============================================================

def load_sidecar_json(model_path: Path) -> Optional[dict]:
    """
    Optional sidecar file.

    Example:
    XGBoost_final.joblib
    XGBoost_final.json

    Can contain:
    {
      "window": 50,
      "feature_cols": ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"],
      "class_order": ["eating", "ruminating", "standing", "walking"]
    }
    """
    sidecar = model_path.with_suffix(".json")

    if not sidecar.exists():
        return None

    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Cannot read sidecar JSON: {sidecar} | {e}")
        return None


# ============================================================
# Prediction helpers
# ============================================================

def get_label_from_prediction(pred_raw: Any, classes: List[str]) -> Tuple[int, str]:
    """
    Convert model prediction into label_id and label string.

    Handles:
    - numeric class id: 0, 1, 2, 3
    - string class label: standing, walking, etc.
    """

    pred = pred_raw[0]

    # If model directly returns string label
    if isinstance(pred, str):
        label = pred
        label_id = classes.index(label) if label in classes else -1
        return label_id, label

    # numpy string type
    if hasattr(pred, "item"):
        try:
            pred_item = pred.item()
            if isinstance(pred_item, str):
                label = pred_item
                label_id = classes.index(label) if label in classes else -1
                return label_id, label
            pred = pred_item
        except Exception:
            pass

    # Numeric class id
    try:
        label_id = int(pred)
        label = classes[label_id] if 0 <= label_id < len(classes) else str(label_id)
        return label_id, label
    except Exception:
        label = str(pred)
        label_id = classes.index(label) if label in classes else -1
        return label_id, label


def get_confidence_and_probs(model: Any, X: np.ndarray, label_id: int) -> Tuple[float, Optional[List[float]]]:
    """
    Return confidence and probability list if model supports predict_proba.
    """
    if not hasattr(model, "predict_proba"):
        return 0.0, None

    try:
        probs = model.predict_proba(X)
        probs = np.array(probs).reshape(-1)
        probs_list = [float(x) for x in probs]

        if 0 <= label_id < len(probs):
            conf = float(probs[label_id])
        else:
            conf = float(np.max(probs))

        return conf, probs_list

    except Exception:
        return 0.0, None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Real-time XGBoost Predictor over MQTT")

    parser.add_argument(
        "--model",
        default=os.getenv("PREDICTOR_MODEL", "deployment_builds/XGBoost_final.joblib"),
        help="Path to XGBoost joblib model."
    )

    parser.add_argument("--broker", default=os.getenv("MQTT_BROKER", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))

    parser.add_argument(
        "--sensors-topic",
        default=os.getenv("MQTT_SENSORS_TOPIC", "farm/cow/+/sensors")
    )

    parser.add_argument(
        "--pred-topic-template",
        default=os.getenv("MQTT_PRED_TOPIC_TMPL", "farm/cow/{cow_id}/predictions")
    )

    parser.add_argument("--qos", type=int, default=int(os.getenv("MQTT_QOS", "0")))

    parser.add_argument(
        "--window",
        type=int,
        default=int(os.getenv("XGB_WINDOW", "50")),
        help="Number of latest sensor rows used to build one feature row."
    )

    parser.add_argument(
        "--pred-every",
        type=int,
        default=int(os.getenv("XGB_PRED_EVERY", "10")),
        help="Run prediction every N new messages per cow."
    )

    parser.add_argument(
        "--classes",
        default=os.getenv("XGB_CLASSES", "eating,ruminating,standing,walking"),
        help="Comma-separated label order. Must match training label order."
    )

    parser.add_argument(
        "--feature-keys",
        default=os.getenv("XGB_FEATURE_KEYS", ""),
        help="Exact payload feature keys. Example: ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"
    )

    parser.add_argument(
        "--feature-mode",
        choices=["auto", "flatten", "stats"],
        default=os.getenv("XGB_FEATURE_MODE", "auto"),
        help="Feature building mode. Use auto normally."
    )

    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Print only, do not publish prediction MQTT."
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise SystemExit(f"[FATAL] XGBoost model not found: {model_path}")

    # Load optional sidecar metadata
    sidecar = load_sidecar_json(model_path)

    if sidecar:
        args.window = int(sidecar.get("window", args.window))

        if isinstance(sidecar.get("feature_cols"), list) and sidecar["feature_cols"]:
            args.feature_keys = ",".join([str(x) for x in sidecar["feature_cols"]])

        if isinstance(sidecar.get("class_order"), list) and sidecar["class_order"]:
            args.classes = ",".join([str(x) for x in sidecar["class_order"]])

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    feature_keys = [k.strip() for k in args.feature_keys.split(",") if k.strip()]

    # Load model
    model = joblib.load(model_path)
    expected_features = get_model_expected_features(model)

    print("[OK] XGBoost / sklearn model loaded")
    print(f"  model            : {model_path}")
    print(f"  expected_features: {expected_features if expected_features is not None else 'unknown'}")
    print(f"  broker           : {args.broker}:{args.port}")
    print(f"  sensors          : {args.sensors_topic}")
    print(f"  pred_tmpl        : {args.pred_topic_template}")
    print(f"  window           : {args.window}")
    print(f"  pred_every       : {args.pred_every}")
    print(f"  classes          : {classes}")
    print(f"  feature_keys     : {feature_keys if feature_keys else 'auto'}")
    print(f"  feature_mode     : {args.feature_mode}")
    print(f"  publish          : {'NO' if args.no_publish else 'YES'}")

    client = make_mqtt_client()

    buffers: Dict[str, Deque[List[float]]] = {}
    counters: Dict[str, int] = {}
    sensor_names_by_cow: Dict[str, List[str]] = {}

    def on_connect(_client, _userdata, _flags, reason_code, properties=None):
        try:
            rc = int(reason_code)
        except Exception:
            rc = 0 if str(reason_code).lower() == "success" else -1

        if rc == 0:
            print("[OK] Connected to MQTT broker")
            _client.subscribe(args.sensors_topic, qos=args.qos)
            print(f"[OK] Subscribed: {args.sensors_topic}")
        else:
            print(f"[ERR] MQTT connection failed rc={reason_code}")

    def on_message(_client, _userdata, msg):
        try:
            raw = msg.payload.decode("utf-8", errors="ignore")
            payload = json.loads(raw)
        except Exception as e:
            print(f"[WARN] Invalid JSON payload from {msg.topic}: {e}")
            return

        if not isinstance(payload, dict):
            return

        cow_id = extract_cow_id_from_topic(msg.topic)

        vec = pick_sensor_vector(payload, feature_keys)

        if vec is None:
            print(f"[WARN] Missing sensor features from topic={msg.topic}")
            return

        # Decide sensor names
        if feature_keys:
            sensor_names = feature_keys
        else:
            if len(vec) == 3:
                sensor_names = ["ax_g", "ay_g", "az_g"]
            elif len(vec) == 6:
                sensor_names = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
            else:
                print(f"[WARN] Unsupported feature length: {len(vec)}")
                return

        if cow_id not in buffers:
            buffers[cow_id] = deque(maxlen=args.window)
            counters[cow_id] = 0
            sensor_names_by_cow[cow_id] = sensor_names
            print(f"[{cow_id}] sensor features = {sensor_names}")

        # Safety: same cow must keep same feature length
        if len(vec) != len(sensor_names_by_cow[cow_id]):
            print(
                f"[WARN] Feature length changed for {cow_id}. "
                f"Expected {len(sensor_names_by_cow[cow_id])}, got {len(vec)}. Skip."
            )
            return

        buffers[cow_id].append(vec)
        counters[cow_id] += 1

        # Need full rolling window first
        if len(buffers[cow_id]) < args.window:
            print(f"[{cow_id}] collecting window {len(buffers[cow_id])}/{args.window}")
            return

        # Predict every N messages
        if args.pred_every > 1 and (counters[cow_id] % args.pred_every) != 0:
            return

        try:
            window_np = np.array(buffers[cow_id], dtype=np.float32)

            X, used_feature_mode = build_model_features(
                window_np=window_np,
                expected_features=expected_features,
                mode=args.feature_mode
            )

            pred_raw = model.predict(X)
            label_id, label = get_label_from_prediction(pred_raw, classes)
            conf, probs_list = get_confidence_and_probs(model, X, label_id)

            print(
                f"[{cow_id}] prediction={label} "
                f"confidence={conf:.3f} "
                f"X_shape={X.shape} "
                f"mode={used_feature_mode}"
            )

            if not args.no_publish:
                pred_topic = args.pred_topic_template.format(cow_id=cow_id)

                out_payload = {
                    "ts": utc_now_iso(),

                    # Main fields for InfluxDB/Grafana
                    "behaviour": label,
                    "predicted_behavior": label,
                    "confidence": conf,

                    # Extra fields
                    "label": label,
                    "label_id": label_id,
                    "model": model_path.name,
                    "model_type": "xgboost",
                    "class_order": classes,
                    "probabilities": probs_list,
                    "window": args.window,
                    "pred_every": args.pred_every,
                    "feature_mode": used_feature_mode,
                    "feature_count": int(X.shape[1]),
                    "features_used": sensor_names_by_cow[cow_id],
                }

                info = _client.publish(pred_topic, json.dumps(out_payload), qos=args.qos)

                try:
                    if info.rc == mqtt.MQTT_ERR_SUCCESS:
                        print(f"[{cow_id}] published -> {pred_topic}")
                    else:
                        print(f"[WARN] Publish rc={info.rc} topic={pred_topic}")
                except Exception:
                    print(f"[{cow_id}] published -> {pred_topic}")

        except Exception as e:
            print(f"[ERR] Prediction error for {cow_id}: {e}")

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.broker, args.port, keepalive=60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Predictor stopped by user")
    except Exception as e:
        raise SystemExit(f"[FATAL] MQTT error: {e}")


if __name__ == "__main__":
    main()
