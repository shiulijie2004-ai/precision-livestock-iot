#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// --- Configuration ---
const char* ssid = "U Mobile_1302_5G";
const char* password = "Pang3828";
const char* mqtt_server = "192.168.0.158";
const int mqtt_port = 1883;

// --- Device Objects ---
WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_MPU6050 mpu;

// --- Function Prototypes ---
void setup_wifi();
void reconnect_mqtt();

void setup() {
    Serial.begin(115200);
    Wire.begin();
    setup_wifi();
    client.setServer(mqtt_server, mqtt_port);

    // Initialize MPU6050
    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip");
        while (1) {
            delay(10);
        }
    }
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    Serial.println("MPU6050 Found!");
}

void loop() {
    if (!client.connected()) {
        reconnect_mqtt();
    }
    client.loop();

    // --- Read Sensor Data ---
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // --- Prepare JSON Payload ---
    char json_payload[200];
    snprintf(json_payload, 200,
             "{\"cow_id\": \"cow_01\", \"ax\": %.2f, \"ay\": %.2f, \"az\": %.2f}",
             a.acceleration.x, a.acceleration.y, a.acceleration.z);

    // --- Publish to MQTT ---
    char topic[50];
    snprintf(topic, 50, "farm/cow/cow_01/sensors");
    client.publish(topic, json_payload);

    Serial.print("Published: ");
    Serial.println(json_payload);

    delay(100); // Corresponds to 10Hz sampling rate
}

// --- Helper Functions ---
void setup_wifi() {
    delay(10);
    Serial.println();
    Serial.print("Connecting to ");
    Serial.println(ssid);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
}

void reconnect_mqtt() {
    while (!client.connected()) {
        Serial.print("Attempting MQTT connection...");
        if (client.connect("ESP32Client-Cow01")) {
            Serial.println("connected");
        } else {
            Serial.print("failed, rc=");
            Serial.print(client.state());
            Serial.println(" try again in 5 seconds");
            delay(5000);
        }
    }
}
