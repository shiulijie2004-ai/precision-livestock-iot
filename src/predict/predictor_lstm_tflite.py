#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor_lstm.py (TFLite-only)

Real-time cattle activity prediction using an exported .tflite model.
This script DOES NOT use any PyTorch (.pth) files, normalizer.npz, or label encoders.

Features:
- Subscribes to an MQTT sensors topic (default: farm/cow/+/sensors)
- Maintains an independent rolling window buffer per cow_id
- Runs inference every N messages (default: pred_every=10)
- Always prints every inference result (even if the label does not change)
- Optionally publishes predictions to farm/cow/{cow_id}/predictions

Payload compatibility:
- Supports keys: ax, ay, az
- Also supports: AccX, AccY, AccZ (case-insensitive)
"""

import os
import json
import argparse
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Deque, List, Optional, Any

import numpy as np
import paho.mqtt.client as mqtt


# ----------------------------
# TFLite interpreter loader
# ----------------------------
def load_tflite_interpreter(model_path: str):
    """
    Load a TFLite Interpreter.

    Tries (in order):
    1) tflite-runtime (recommended for lightweight deployment)
    2) tensorflow (fallback if you already have TF installed)
    """
    last_err = None

    # Prefer tflite-runtime
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        itp = Interpreter(model_path=model_path)
        itp.allocate_tensors()
        return itp
    except Exception as e:
        last_err = e

    # Fallback to tensorflow
    try:
        try:
            from tensorflow.lite import Interpreter  # type: ignore
        except Exception:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

        itp = Interpreter(model_path=model_path)
        itp.allocate_tensors()
        return itp
    except Exception as e:
        last_err = e

    raise RuntimeError(
        "Cannot load .tflite. Install either `tflite-runtime` or `tensorflow`.\n"
        f"Last error: {last_err}"
    )


# ----------------------------
# MQTT helpers
# ----------------------------
def make_mqtt_client():
    """
    Create a Paho MQTT client compatible with both old and new callback API versions.
    """
    try:
        return mqtt.Client(
            client_id="prediction_lstm_tflite",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
    except Exception:
        return mqtt.Client(client_id="prediction_lstm_tflite")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_cow_id_from_topic(topic: str) -> str:
    """
    Extract cow_id from topic.

    Supported patterns:
    - farm/cow/<cow_id>/sensors
    - farm/cow/<cow_id>/anything/...
    """
    parts = [p for p in topic.split("/") if p]
    for i, p in enumerate(parts):
        if p.lower() == "cow" and i + 1 < len(parts):
            return parts[i + 1]
    # Fallback: best-effort index
    return parts[2] if len(parts) >= 3 else "unknown"


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def pick_features_from_payload(payload: dict, feature_keys: List[str], expected_F: int) -> Optional[List[float]]:
    """
    Extract a feature vector from the JSON payload.

    If feature_keys is provided, use that exact order.
    Otherwise, auto-detect 3-axis accelerometer keys:
      - ax/ay/az (preferred)
      - AccX/AccY/AccZ (fallback)

    Returns:
      - List[float] of length expected_F, or None if missing / mismatch.
    """
    # Build a case-insensitive key map
    lower_map = {str(k).lower(): k for k in payload.keys()}

    def getv(key_candidates: List[str]) -> Optional[float]:
        for k in key_candidates:
            kk = k.lower()
            if kk in lower_map:
                raw_key = lower_map[kk]
                return _safe_float(payload.get(raw_key))
        return None

    if feature_keys:
        vec: List[float] = []
        for k in feature_keys:
            v = getv([k])
            if v is None:
                return None
            vec.append(v)
        return vec if len(vec) == expected_F else None

    # Auto mode: only supports 3-axis
    ax = getv(["ax", "accx"])
    ay = getv(["ay", "accy"])
    az = getv(["az", "accz"])
    if ax is None or ay is None or az is None:
        return None

    if expected_F != 3:
        # You said all models have consistent F; refuse to guess extra features.
        return None

    return [ax, ay, az]


def load_sidecar_json(model_path: Path) -> Optional[dict]:
    """
    Optional: read a JSON sidecar with the same basename as the .tflite.
    Example:
      LSTM_final.tflite
      LSTM_final.json

    If present, we can override:
      - window
      - feature_cols (used as feature_keys)
      - class_order
    """
    sidecar = model_path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-time LSTM TFLite Predictor (MQTT)")
    parser.add_argument(
        "--model",
        default=os.getenv("PREDICTOR_MODEL", os.getenv("LSTM_TFLITE", "deployment_builds/LSTM_final.tflite")),
        help="Path to the .tflite model, e.g. deployment_builds/LSTM_final.tflite",
    )
    parser.add_argument("--broker", default=os.getenv("MQTT_BROKER", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--sensors-topic", default=os.getenv("MQTT_SENSORS_TOPIC", "farm/cow/+/sensors"))
    parser.add_argument("--pred-topic-template", default=os.getenv("MQTT_PRED_TOPIC_TMPL", "farm/cow/{cow_id}/predictions"))
    parser.add_argument("--qos", type=int, default=int(os.getenv("MQTT_QOS", "0")))
    parser.add_argument("--window", type=int, default=int(os.getenv("LSTM_WINDOW", "50")))
    parser.add_argument("--features", type=int, default=int(os.getenv("LSTM_FEATURES", "3")))
    parser.add_argument("--pred-every", type=int, default=int(os.getenv("LSTM_PRED_EVERY", "10")))
    parser.add_argument(
        "--classes",
        default=os.getenv("LSTM_CLASSES", "eating,ruminating,standing,walking"),
        help="Comma-separated label order for model outputs.",
    )
    parser.add_argument(
        "--feature-keys",
        default=os.getenv("LSTM_FEATURE_KEYS", ""),
        help="Comma-separated payload keys in the exact input order (optional). Example: ax,ay,az",
    )
    parser.add_argument("--no-publish", action="store_true", help="Do not publish predictions (print only).")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"[FATAL] TFLite model not found: {model_path}")

    # Optional sidecar overrides (if you have them)
    sidecar = load_sidecar_json(model_path)
    if sidecar:
        args.window = int(sidecar.get("window", args.window))
        if isinstance(sidecar.get("feature_cols"), list) and sidecar["feature_cols"]:
            args.feature_keys = ",".join([str(x) for x in sidecar["feature_cols"]])
        if isinstance(sidecar.get("class_order"), list) and sidecar["class_order"]:
            args.classes = ",".join([str(x) for x in sidecar["class_order"]])

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    feature_keys = [k.strip() for k in args.feature_keys.split(",") if k.strip()]

    interpreter = load_tflite_interpreter(str(model_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if not input_details:
        raise SystemExit("[FATAL] No input tensors found in this .tflite.")
    if not output_details:
        raise SystemExit("[FATAL] No output tensors found in this .tflite.")

    in0 = input_details[0]
    out0 = output_details[0]

    in_index = int(in0["index"])
    out_index = int(out0["index"])
    in_shape = list(in0.get("shape", []))
    in_dtype = in0.get("dtype", np.float32)

    # Infer expected (T, F) from model input shape
    # Common shapes: [1, T, F] or [T, F]
    T_expected: Optional[int] = None
    F_expected: Optional[int] = None
    if len(in_shape) == 3:
        # [batch, T, F]
        if in_shape[1] != -1:
            T_expected = int(in_shape[1])
        if in_shape[2] != -1:
            F_expected = int(in_shape[2])
    elif len(in_shape) == 2:
        if in_shape[0] != -1:
            T_expected = int(in_shape[0])
        if in_shape[1] != -1:
            F_expected = int(in_shape[1])

    # Safety checks: fail fast if mismatch
    if T_expected is not None and T_expected != args.window:
        raise SystemExit(f"[FATAL] Model expects window={T_expected}, but you set window={args.window}.")
    if F_expected is not None and F_expected != args.features:
        raise SystemExit(f"[FATAL] Model expects features={F_expected}, but you set features={args.features}.")

    # Output classes count alignment (if model has fixed C)
    out_shape = list(out0.get("shape", []))
    C_expected: Optional[int] = None
    if out_shape and out_shape[-1] != -1:
        C_expected = int(out_shape[-1])
    if C_expected is not None and len(classes) != C_expected:
        classes = [f"class_{i}" for i in range(C_expected)]

    print(f"[OK] TFLite model loaded: {model_path}")
    print(f"  broker      : {args.broker}:{args.port}")
    print(f"  sensors     : {args.sensors_topic}")
    print(f"  pred_tmpl   : {args.pred_topic_template}")
    print(f"  window      : {args.window}")
    print(f"  features    : {args.features}")
    print(f"  pred_every  : {args.pred_every}")
    print(f"  classes     : {classes}")
    if feature_keys:
        print(f"  feature_keys: {feature_keys}")
    print(f"  publish     : {'NO' if args.no_publish else 'YES'}")

    client = make_mqtt_client()

    buffers: Dict[str, Deque[List[float]]] = {}
    counters: Dict[str, int] = {}

    def on_connect(_client, _userdata, _flags, reason_code, properties=None):
        if reason_code == 0:
            print("[OK] Connected")
            _client.subscribe(args.sensors_topic, qos=args.qos)
            print(f"[OK] Subscribed: {args.sensors_topic}")
        else:
            print(f"[ERR] Connection failed rc={reason_code}")

    def on_message(_client, _userdata, msg):
        # Parse JSON payload
        try:
            payload = json.loads(msg.payload.decode("utf-8", errors="ignore"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        cow_id = extract_cow_id_from_topic(msg.topic)

        if cow_id not in buffers:
            buffers[cow_id] = deque(maxlen=args.window)
            counters[cow_id] = 0

        vec = pick_features_from_payload(payload, feature_keys, args.features)
        if vec is None:
            return

        buffers[cow_id].append(vec)
        counters[cow_id] += 1

        # Need a full window before inference
        if len(buffers[cow_id]) < args.window:
            return

        # Throttle inference
        if args.pred_every > 1 and (counters[cow_id] % args.pred_every) != 0:
            return

        try:
            x = np.array(buffers[cow_id], dtype=np.float32)  # [T, F]

            # Shape to match model expectation
            if len(in_shape) == 3:
                x = x.reshape(1, args.window, args.features)  # [1, T, F]
            else:
                x = x.reshape(args.window, args.features)     # [T, F]

            # Handle quantized or float input
            if in_dtype in (np.int8, np.uint8):
                q = in0.get("quantization_parameters", {}) or {}
                scales = q.get("scales", [])
                zero_points = q.get("zero_points", [])
                if len(scales) and len(zero_points) and float(scales[0]) != 0.0:
                    scale = float(scales[0])
                    zp = int(zero_points[0])
                    xq = np.rint(x / scale + zp)
                    if in_dtype == np.int8:
                        xq = np.clip(xq, -128, 127).astype(np.int8)
                    else:
                        xq = np.clip(xq, 0, 255).astype(np.uint8)
                    interpreter.set_tensor(in_index, xq)
                else:
                    interpreter.set_tensor(in_index, x.astype(in_dtype))
            else:
                interpreter.set_tensor(in_index, x.astype(in_dtype))

            interpreter.invoke()
            y = interpreter.get_tensor(out_index)
            y = np.array(y).reshape(-1)  # [C]

            label_id = int(np.argmax(y)) if y.size else 0
            conf = float(y[label_id]) if (y.size and 0 <= label_id < y.size) else 0.0
            label = classes[label_id] if 0 <= label_id < len(classes) else str(label_id)

            # ALWAYS PRINT (even if label did not change)
            print(f"[{cow_id}] Current predicted activity: {label} (conf={conf:.3f})")

            if not args.no_publish:
                pred_topic = args.pred_topic_template.format(cow_id=cow_id)
                out_payload = {
                    "model": model_path.name,
                    "label": label,
                    "confidence": conf,
                    "label_id": label_id,
                    "ts": utc_now_iso(),
                }
                _client.publish(pred_topic, json.dumps(out_payload), qos=args.qos)

        except Exception as e:
            print("[ERR] Inference error:", e)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(args.broker, args.port, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
