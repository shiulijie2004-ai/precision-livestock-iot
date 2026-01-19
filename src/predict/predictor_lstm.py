# src/predict/predictor_lstm.py
# Real-time LSTM predictor (ALWAYS PRINT)
# - Subscribes to MQTT sensors topic (default: farm/cow/+/sensors)
# - Per-cow fixed window buffering (default 50 samples)
# - Throttled inference (default every 10 sensor msgs ~ 1s @10Hz)
# - Publishes prediction to farm/cow/<cow_id>/predictions as JSON:
#   {"model":"lstm","label":"...","confidence":0-1,"label_id":int,"ts":ISO}
# - ALWAYS prints each prediction (even if same label)

import os
import sys
import json
import argparse
from collections import deque
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
import joblib
import paho.mqtt.client as mqtt

# ---------- Path bootstrap ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.models.lstm import CowActivityLSTM
except Exception as e:
    raise SystemExit(f"[FATAL] Cannot import CowActivityLSTM from src.models.lstm: {e}")

# ---------- Env defaults ----------
BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT   = int(os.getenv("MQTT_PORT", "1883"))

SENSORS_TOPIC = os.getenv("MQTT_SENSORS_TOPIC", "farm/cow/+/sensors")
PRED_TOPIC_TEMPLATE = os.getenv("MQTT_PRED_TOPIC_TEMPLATE", "farm/cow/{cow_id}/predictions")

CHECKPOINT_PATH = Path(os.getenv("LSTM_CHECKPOINT", str(PROJECT_ROOT / "results" / "checkpoints" / "lstm_best_model.pth")))
ENCODER_PATH    = Path(os.getenv("LSTM_ENCODER",    str(PROJECT_ROOT / "results" / "checkpoints" / "label_encoder.joblib")))
NORMALIZER_PATH = Path(os.getenv("LSTM_NORMALIZER", str(PROJECT_ROOT / "results" / "checkpoints" / "normalizer.npz")))

DEVICE      = os.getenv("LSTM_DEVICE", "cpu")
INPUT_SIZE  = int(os.getenv("LSTM_INPUT_SIZE", "3"))
HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", "64"))
NUM_LAYERS  = int(os.getenv("LSTM_NUM_LAYERS", "2"))

WINDOW_SIZE = int(os.getenv("LSTM_WINDOW", "50"))
PRED_EVERY  = int(os.getenv("LSTM_PRED_EVERY", "10"))  # throttle

# ---------- Helpers ----------
def _make_mqtt_client():
    try:
        return mqtt.Client(
            client_id="prediction_lstm",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
    except Exception:
        return mqtt.Client(client_id="prediction_lstm")

def _cow_id_from_topic(topic: str) -> str:
    parts = topic.split("/")
    if len(parts) >= 4 and parts[0] == "farm" and parts[1] == "cow":
        return parts[2]
    return "unknown"

def _pred_topic_for_cow(cow_id: str, template: str) -> str:
    try:
        return template.format(cow_id=cow_id)
    except Exception:
        return f"farm/cow/{cow_id}/predictions"

def _load_checkpoint_state(ckpt_path: Path):
    obj = torch.load(str(ckpt_path), map_location=DEVICE)
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj

def _load_normalizer(npz_path: Path):
    if not npz_path.exists():
        return None, None
    try:
        norm = np.load(str(npz_path))
        mean = norm.get("mean")
        std  = norm.get("std")
        if mean is None or std is None:
            return None, None
        mean = np.array(mean, dtype=np.float32).reshape(-1)
        std  = np.array(std, dtype=np.float32).reshape(-1)
        std[std == 0] = 1.0
        return mean, std
    except Exception as e:
        print(f"[warn] failed to load normalizer.npz: {e}")
        return None, None

# ---------- Load encoder & model ----------
if not ENCODER_PATH.exists():
    raise SystemExit(f"[FATAL] Label encoder not found: {ENCODER_PATH}")

le = joblib.load(str(ENCODER_PATH))
classes = [str(c) for c in list(getattr(le, "classes_", []))]
NUM_CLASSES = len(classes)
if NUM_CLASSES <= 0:
    raise SystemExit("[FATAL] LabelEncoder has no classes_.")

if not CHECKPOINT_PATH.exists():
    raise SystemExit(f"[FATAL] LSTM checkpoint not found: {CHECKPOINT_PATH}")

model = CowActivityLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
state = _load_checkpoint_state(CHECKPOINT_PATH)
model.load_state_dict(state)
model.eval()
print("[ok] LSTM model loaded:", CHECKPOINT_PATH)

MEAN, STD = _load_normalizer(NORMALIZER_PATH)
if MEAN is not None and STD is not None:
    print("[ok] Loaded training normalizer:", NORMALIZER_PATH)
else:
    print("[warn] normalizer.npz not found/invalid; running WITHOUT normalization.")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="LSTM real-time predictor (always print)")
    parser.add_argument("--broker", default=BROKER)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--sensors-topic", default=SENSORS_TOPIC)
    parser.add_argument("--pred-topic-template", default=PRED_TOPIC_TEMPLATE)
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--pred-every", type=int, default=PRED_EVERY)
    args = parser.parse_args()

    pred_template = args.pred_topic_template

    # Per-cow state (prevents mixing different cows into one window)
    buffers = {}   # cow_id -> deque
    counters = {}  # cow_id -> msg_count

    print("[ok] LSTM predictor ready")
    print(f"  broker     : {args.broker}:{args.port}")
    print(f"  sensors    : {args.sensors_topic}")
    print(f"  pred_tmpl  : {pred_template}")
    print(f"  window     : {args.window}")
    print(f"  pred_every : {args.pred_every}")
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
        # parse JSON safely
        try:
            payload = json.loads(msg.payload.decode("utf-8", errors="ignore"))
        except Exception:
            return

        if not all(k in payload for k in ("ax", "ay", "az")):
            return

        try:
            ax = float(payload["ax"])
            ay = float(payload["ay"])
            az = float(payload["az"])
        except Exception:
            return

        cow_id = _cow_id_from_topic(msg.topic)
        if cow_id not in buffers:
            buffers[cow_id] = deque(maxlen=args.window)
            counters[cow_id] = 0

        buffers[cow_id].append([ax, ay, az])

        if len(buffers[cow_id]) < args.window:
            return

        counters[cow_id] += 1
        if args.pred_every > 1 and (counters[cow_id] % args.pred_every != 0):
            return

        pred_topic = _pred_topic_for_cow(cow_id, pred_template)

        try:
            window = np.array(buffers[cow_id], dtype=np.float32)  # (T,3)

            if MEAN is not None and STD is not None and MEAN.size >= 3 and STD.size >= 3:
                window = (window - MEAN[:3]) / STD[:3]

            x = torch.from_numpy(window).float().unsqueeze(0).to(DEVICE)  # (1,T,3)

            with torch.no_grad():
                logits = model(x)  # (1,C)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                conf = float(np.max(probs))
                label = str(le.inverse_transform([pred_idx])[0])

            out = {
                "model": "lstm",
                "label": label,
                "confidence": conf,
                "label_id": pred_idx,
                "ts": datetime.now(timezone.utc).isoformat()
            }

            client.publish(pred_topic, json.dumps(out), qos=0, retain=False)

            # ✅ ALWAYS PRINT (even if same label)
            print(f"[{cow_id}] Current predicted activity: {label} (conf={conf:.3f})")

        except Exception as e:
            print("[err] predict:", e)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(args.broker, args.port, keepalive=60)
    client.loop_forever()

if __name__ == "__main__":
    main()

