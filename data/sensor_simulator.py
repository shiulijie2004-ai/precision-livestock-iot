#!/usr/bin/env python3
"""
sensor_simulator_updated.py

Publishes simulated tri-axial accelerometer (+ optional temperature) telemetry to MQTT,
so you can test your full pipeline (MQTT -> mqtt_consumer -> InfluxDB -> Grafana)
without hardware.

Default topic pattern matches your pipeline wildcard:
  farm/cow/{cow_id}/sensors

Payload is JSON and includes:
  - cow_id
  - t_sensor (UTC ISO8601 timestamp, "Z")
  - msg_counter (monotonic per cow)
  - ax, ay, az (float)
  - temp (optional, float, Celsius)
  - sim_activity (optional, string)

You can also simulate packet drops + extra delays to create missingness/latency.
"""

import os
import json
import time
import math
import random
from datetime import datetime, timezone
from typing import Dict, List

import paho.mqtt.client as mqtt


def utc_iso_now() -> str:
    # Example: "2025-12-22T04:12:34.123Z"
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def accel_for_activity(activity: str, t: float) -> Dict[str, float]:
    """
    Very simple IMU simulation:
      - standing: near-zero noise
      - ruminating: small, regular oscillations
      - grazing: moderate irregular motion
      - walking: larger periodic motion
    """
    # Base noise level
    n = random.gauss(0.0, 0.02)

    if activity == "standing":
        ax = 0.02 * random.gauss(0, 1)
        ay = 0.02 * random.gauss(0, 1)
        az = 0.98 + 0.02 * random.gauss(0, 1)  # pretend gravity component
    elif activity == "ruminating":
        ax = 0.04 * math.sin(2 * math.pi * 0.8 * t) + 0.02 * random.gauss(0, 1)
        ay = 0.03 * math.sin(2 * math.pi * 0.9 * t + 0.7) + 0.02 * random.gauss(0, 1)
        az = 0.98 + 0.02 * math.sin(2 * math.pi * 0.6 * t) + 0.02 * random.gauss(0, 1)
    elif activity == "grazing":
        ax = 0.10 * math.sin(2 * math.pi * 1.2 * t) + 0.04 * random.gauss(0, 1)
        ay = 0.08 * math.sin(2 * math.pi * 1.0 * t + 1.3) + 0.04 * random.gauss(0, 1)
        az = 0.98 + 0.04 * math.sin(2 * math.pi * 0.8 * t + 0.2) + 0.04 * random.gauss(0, 1)
    else:  # walking
        ax = 0.22 * math.sin(2 * math.pi * 1.8 * t) + 0.06 * random.gauss(0, 1)
        ay = 0.18 * math.sin(2 * math.pi * 1.7 * t + 0.9) + 0.06 * random.gauss(0, 1)
        az = 0.98 + 0.10 * math.sin(2 * math.pi * 1.6 * t + 0.1) + 0.06 * random.gauss(0, 1)

    # A tiny shared noise bump to keep streams "alive"
    ax += n
    ay += n * 0.8
    az += n * 0.3

    return {"ax": float(ax), "ay": float(ay), "az": float(az)}


def temp_for_activity(activity: str, t: float, base: float, fever: bool) -> float:
    """
    Temperature simulation (Celsius).
    This is deliberately simple, but good enough for:
      - dashboard trend plots
      - deviation score logic
      - alert annotations
    """
    # Typical dairy cow temp ballpark ~38-39C; keep within safe bounds for simulation.
    offsets = {
        "standing": -0.10,
        "ruminating": -0.20,
        "grazing": 0.00,
        "walking": 0.25,
    }
    circadian = 0.08 * math.sin(2 * math.pi * (t / 3600.0) / 6.0)  # slow drift
    noise = random.gauss(0.0, 0.05)

    temp = base + offsets.get(activity, 0.0) + circadian + noise

    if fever:
        # Add a mild sustained elevation (still clamped)
        temp += 0.7 + 0.15 * math.sin(2 * math.pi * 0.03 * t)

    return float(clamp(temp, 36.5, 41.5))


def choose_activity(mode: str, activities: List[str], elapsed_s: float, last: str) -> str:
    if mode == "fixed":
        return last
    if mode == "random":
        return random.choice(activities)
    # default: cycle
    period = float(os.getenv("ACTIVITY_CYCLE_SEC", "30"))
    idx = int(elapsed_s // period) % len(activities)
    return activities[idx]


def main() -> None:
    broker = os.getenv("MQTT_BROKER", "localhost")
    port = int(os.getenv("MQTT_PORT", "1883"))
    topic_tpl = os.getenv("TOPIC_TEMPLATE", "farm/cow/{cow_id}/sensors")

    cows = parse_csv_list(os.getenv("COW_IDS", "cow_01"))
    if not cows:
        cows = ["cow_01"]

    publish_interval = float(os.getenv("PUBLISH_INTERVAL_SEC", "1.0"))

    # Optional sim knobs
    include_temp = os.getenv("INCLUDE_TEMP", "1").lower() in ("1", "true", "yes", "y")
    include_activity = os.getenv("INCLUDE_ACTIVITY", "1").lower() in ("1", "true", "yes", "y")

    activity_mode = os.getenv("ACTIVITY_MODE", "cycle").lower()  # cycle|random|fixed
    activities = parse_csv_list(os.getenv("ACTIVITIES", "standing,ruminating,grazing,walking"))
    if not activities:
        activities = ["standing", "ruminating", "grazing", "walking"]

    drop_prob = float(os.getenv("DROP_PROB", "0.02"))  # simulate missing packets
    extra_delay_prob = float(os.getenv("EXTRA_DELAY_PROB", "0.05"))  # simulate bursts/latency
    extra_delay_max = float(os.getenv("EXTRA_DELAY_MAX_SEC", "2.0"))
    duplicate_prob = float(os.getenv("DUPLICATE_PROB", "0.01"))  # resend last payload sometimes

    fever_prob = float(os.getenv("FEVER_PROB", "0.01"))  # probability to enter fever state
    fever_min_sec = float(os.getenv("FEVER_MIN_SEC", "60"))
    fever_max_sec = float(os.getenv("FEVER_MAX_SEC", "180"))

    # Per-cow state
    counters = {c: 0 for c in cows}
    last_payload: Dict[str, Dict] = {c: {} for c in cows}
    last_activity = {c: activities[0] for c in cows}

    # Temperature baseline per cow
    base_temp = {c: 38.6 + random.uniform(-0.2, 0.2) for c in cows}
    fever_until = {c: 0.0 for c in cows}

    client_id = f"sensor_sim_{os.getpid()}_{random.randint(1000,9999)}"
    client = mqtt.Client(client_id=client_id)
    client.connect(broker, port, keepalive=60)

    print(f"[sim] connected to mqtt://{broker}:{port}")
    print(f"[sim] cows={cows} interval={publish_interval}s topic_template='{topic_tpl}'")
    print(f"[sim] include_temp={include_temp} include_activity={include_activity}")
    print(f"[sim] drop_prob={drop_prob} extra_delay_prob={extra_delay_prob} duplicate_prob={duplicate_prob}")

    start = time.time()
    next_tick = start

    while True:
        now = time.time()
        if now < next_tick:
            time.sleep(max(0.0, next_tick - now))
            continue

        elapsed = now - start
        # Publish one message per cow each tick
        for cow_id in cows:
            # Simulate extra delay bursts (creates end-to-end lag)
            if random.random() < extra_delay_prob:
                time.sleep(random.random() * extra_delay_max)

            # Simulate packet drop (creates missingness)
            if random.random() < drop_prob:
                # skip publishing entirely
                continue

            # Simulate duplicates
            if last_payload.get(cow_id) and random.random() < duplicate_prob:
                payload = last_payload[cow_id]
            else:
                activity = choose_activity(activity_mode, activities, elapsed, last_activity[cow_id])
                last_activity[cow_id] = activity

                counters[cow_id] += 1
                t_sensor = utc_iso_now()

                # Generate accel
                accel = accel_for_activity(activity, elapsed)

                payload = {
                    "cow_id": cow_id,
                    "t_sensor": t_sensor,
                    "msg_counter": counters[cow_id],
                    "ax": accel["ax"],
                    "ay": accel["ay"],
                    "az": accel["az"],
                }

                # Temperature (optional)
                if include_temp:
                    # Enter fever state sometimes (sustained)
                    if fever_until[cow_id] <= now and random.random() < fever_prob:
                        fever_until[cow_id] = now + random.uniform(fever_min_sec, fever_max_sec)

                    fever = now < fever_until[cow_id]
                    payload["temp"] = temp_for_activity(activity, elapsed, base_temp[cow_id], fever)

                if include_activity:
                    payload["sim_activity"] = activity

                last_payload[cow_id] = payload

            topic = topic_tpl.format(cow_id=cow_id)
            client.publish(topic, json.dumps(payload), qos=0, retain=False)

            # Lightweight console output
            if payload.get("msg_counter", 0) % 10 == 0:
                extras = []
                if "temp" in payload:
                    extras.append(f"temp={payload['temp']:.2f}")
                if "sim_activity" in payload:
                    extras.append(f"act={payload['sim_activity']}")
                extra_s = " ".join(extras)
                print(f"[{cow_id}] #{payload.get('msg_counter')} ax={payload.get('ax'):.3f} ay={payload.get('ay'):.3f} az={payload.get('az'):.3f} {extra_s}")

        # Keep the MQTT network loop healthy
        client.loop(timeout=0.01)

        next_tick += publish_interval


if __name__ == "__main__":
    main()
