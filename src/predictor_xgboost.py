#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor_xgboost.py

Correct real-time XGBoost predictor for Japan cow accelerometer-only model.

This version is designed for:

Training dataset:
    japan_cows_1to6_merged_dashboard.csv

Training features:
    AccX, AccY, AccZ

Live MQTT sensor keys:
    ax_g, ay_g, az_g

Model:
    deployment_builds/XGBoost_final.joblib

Expected model input:
    18 statistical features

Feature order MUST match train_correct_xgboost_japan.py:

    mean_ax, mean_ay, mean_az,
    std_ax, std_ay, std_az,
    min_ax, min_ay, min_az,
    max_ax, max_ay, max_az,
    range_ax, range_ay, range_az,
    energy_ax, energy_ay, energy_az

Important:
    energy = sum of squares over the window, NOT mean of squares.

Pipeline:
    MQTT sensor topic:
        farm/cow/+/sensors

    Prediction output topic:
        farm/cow/{cow_id}/predictions
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
    Create a Paho MQTT client compatible with old and new paho-mqtt versions.
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

    Examples:
        ax_g, ax, acc_x, AccX
        ay_g, ay, acc_y, AccY
        az_g, az, acc_z, AccZ
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

    For your Japan cow XGBoost model, use:
        --feature-keys ax_g,ay_g,az_g

    Then the vector will be:
        [ax_g, ay_g, az_g]

    Gyro will NOT be used unless you explicitly include:
        gx_dps,gy_dps,gz_dps
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

    # Fallback auto-detection: accelerometer first
    ax = get_value_case_insensitive(payload, ["ax_g", "ax", "accx", "acc_x", "AccX"])
    ay = get_value_case_insensitive(payload, ["ay_g", "ay", "accy", "acc_y", "AccY"])
    az = get_value_case_insensitive(payload, ["az_g", "az", "accz", "acc_z", "AccZ"])

    if ax is None or ay is None or az is None:
        return None

    # Default for your project: accelerometer-only
    return [ax, ay, az]


# ============================================================
# Feature builders
# ============================================================

def build_flatten_features(window_data: np.ndarray) -> np.ndarray:
    """
    Flatten raw window.

    Example:
        window = 50
        sensors = 3

    Output:
        50 x 3 = 150 features

    Your current Japan cow XGBoost model should NOT use this,
    because your correct model expects 18 stats features.
    """
    return window_data.astype(np.float32).reshape(1, -1)


def build_stat_features(window_data: np.ndarray) -> np.ndarray:
    """
    Build statistical features in the SAME order as your training code.

    Input:
        window_data shape = [window, sensors]

    For Japan cow:
        window_data shape = [50, 3]
        sensor order = ax_g, ay_g, az_g

    Output order:
        mean_ax, mean_ay, mean_az,
        std_ax, std_ay, std_az,
        min_ax, min_ay, min_az,
        max_ax, max_ay, max_az,
        range_ax, range_ay, range_az,
        energy_ax, energy_ay, energy_az

    Important:
        energy = sum of squares over the window
        same as training code:
            (df_x * df_x).rolling(...).sum()
    """
    X = np.asarray(window_data, dtype=np.float32)

    mean = np.mean(X, axis=0).astype(np.float32)
    std = np.std(X, axis=0, ddof=0).astype(np.float32)
    min_v = np.min(X, axis=0).astype(np.float32)
    max_v = np.max(X, axis=0).astype(np.float32)
    range_v = (max_v - min_v).astype(np.float32)
    energy = np.sum(X ** 2, axis=0).astype(np.float32)

    feats = np.concatenate([
        mean,
        std,
        min_v,
        max_v,
        range_v,
        energy,
    ]).astype(np.float32)

    return feats.reshape(1, -1)


def get_model_expected_features(model: Any) -> Optional[int]:
    """
    Try to detect number of features expected by sklearn / XGBoost model.
    """
    if hasattr(model, "n_features_in_"):
        try:
            return int(model.n_features_in_)
        except Exception:
            pass

    if hasattr(model, "named_steps"):
        try:
            for step in reversed(list(model.named_steps.values())):
                if hasattr(step, "n_features_in_"):
                    return int(step.n_features_in_)
        except Exception:
            pass

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
        auto
        flatten
        stats

    For your correct Japan cow model:
        expected_features = 18
        window = 50
        feature_keys = ax_g, ay_g, az_g

    Therefore:
        flatten = 50 x 3 = 150
        stats = 3 x 6 = 18

    Auto will choose stats.
    """

    flat_X = build_flatten_features(window_np)
    stat_X = build_stat_features(window_np)

    if mode == "flatten":
        return flat_X, "flatten"

    if mode == "stats":
        return stat_X, "stats"

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

    return stat_X, "stats"


# ============================================================
# Sidecar metadata
# ============================================================

def load_sidecar_json(model_path: Path) -> Optional[dict]:
    """
    Optional sidecar file.

    Supported:
        XGBoost_final.json
        XGBoost_final_features.json

    It can contain either:
        feature_cols

    or:
        live_sensor_keys_expected
    """
    candidates = [
        model_path.with_suffix(".json"),
        model_path.with_name(model_path.stem + "_features.json"),
    ]

    for sidecar in candidates:
        if not sidecar.exists():
            continue

        try:
            print(f"[OK] Found metadata JSON: {sidecar}")
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
    """

    pred = pred_raw[0]

    if isinstance(pred, str):
        label = pred
        label_id = classes.index(label) if label in classes else -1
        return label_id, label

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

    except Exception as e:
        print(f"[WARN] predict_proba failed: {e}")
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
        help="Exact payload feature keys. For Japan cow use: ax_g,ay_g,az_g"
    )

    parser.add_argument(
        "--feature-mode",
        choices=["auto", "flatten", "stats"],
        default=os.getenv("XGB_FEATURE_MODE", "auto"),
        help="Use stats for the correct Japan cow XGBoost model."
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

    # Load optional metadata
    sidecar = load_sidecar_json(model_path)

    if sidecar:
        if "window" in sidecar:
            args.window = int(sidecar.get("window", args.window))

        if isinstance(sidecar.get("feature_cols"), list) and sidecar["feature_cols"]:
            args.feature_keys = ",".join([str(x) for x in sidecar["feature_cols"]])

        if isinstance(sidecar.get("live_sensor_keys_expected"), list) and sidecar["live_sensor_keys_expected"]:
            args.feature_keys = ",".join([str(x) for x in sidecar["live_sensor_keys_expected"]])

        if isinstance(sidecar.get("class_order"), list) and sidecar["class_order"]:
            args.classes = ",".join([str(x) for x in sidecar["class_order"]])

        if isinstance(sidecar.get("target_classes"), list) and sidecar["target_classes"]:
            args.classes = ",".join([str(x) for x in sidecar["target_classes"]])

        if "feature_mode" in sidecar:
            args.feature_mode = str(sidecar.get("feature_mode", args.feature_mode))

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    feature_keys = [k.strip() for k in args.feature_keys.split(",") if k.strip()]

    if not classes:
        raise SystemExit("[FATAL] No classes provided.")

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

    # Safety warning for your case
    if expected_features == 18 and len(feature_keys) == 3:
        print("[OK] Japan cow accelerometer-only setup detected: 3-axis stats -> 18 features")

    if expected_features == 300:
        print("[WARN] This model expects 300 features. That usually means 50 x 6-axis flatten.")
        print("[WARN] This is NOT suitable for Japan cow accelerometer-only prediction.")

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
            print(f"[WARN] Payload keys: {list(payload.keys())}")
            print(f"[WARN] Required feature_keys: {feature_keys if feature_keys else 'auto ax_g/ay_g/az_g'}")
            return

        if feature_keys:
            sensor_names = feature_keys
        else:
            sensor_names = ["ax_g", "ay_g", "az_g"]

        if cow_id not in buffers:
            buffers[cow_id] = deque(maxlen=args.window)
            counters[cow_id] = 0
            sensor_names_by_cow[cow_id] = sensor_names
            print(f"[{cow_id}] sensor features = {sensor_names}")

        if len(vec) != len(sensor_names_by_cow[cow_id]):
            print(
                f"[WARN] Feature length changed for {cow_id}. "
                f"Expected {len(sensor_names_by_cow[cow_id])}, got {len(vec)}. Skip."
            )
            return

        buffers[cow_id].append(vec)
        counters[cow_id] += 1

        if len(buffers[cow_id]) < args.window:
            print(f"[{cow_id}] collecting window {len(buffers[cow_id])}/{args.window}")
            return

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
