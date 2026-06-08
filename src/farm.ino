/* =====================================================
   HELTEC WiFi LoRa 32 V4 - OTAA SENSOR PAYLOAD TX
   Sensors: MPU6050 + DS18B20
   LoRaWAN: OTAA, AS923, Class A

   Payload: 20 bytes

   Payload Format:
   byte 0-1   payload counter
   byte 2     status: bit0=mpu_ok, bit1=ds18b20_ok
   byte 3     reserved
   byte 4-5   ax_mg
   byte 6-7   ay_mg
   byte 8-9   az_mg
   byte 10-11 gx_cdps   // gyro dps * 100
   byte 12-13 gy_cdps
   byte 14-15 gz_cdps
   byte 16-17 mpu_temp_centi_c
   byte 18-19 ds18b20_temp_centi_c

   Important:
   - Use DR3 or higher because 20 bytes is larger than your old payload.
   - TTS decoder must also expect 20 bytes.
   ===================================================== */

#include "LoRaWan_APP.h"
#include "Arduino.h"
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "esp_system.h"

// =====================================================
// DEBUG SETTINGS
// =====================================================
#define DEBUG_SENSOR_PRINT 0
#define DEBUG_PAYLOAD_PRINT 1
#define DEBUG_SEND_PRINT 0

// =====================================================
// HELTEC LICENSE
// Keep your existing license.
// =====================================================
uint32_t license[4] = {
  0xBCF9D049,
  0x065CADF6,
  0xAF3EB5BA,
  0x1C5FC357
};

// =====================================================
// OTAA KEYS - MUST MATCH TTS EXACTLY
// Keep your existing DevEUI / JoinEUI / AppKey.
// =====================================================
uint8_t devEui[] = {
  0x00, 0x00, 0x30, 0xB9,
  0xA3, 0x1B, 0x5B, 0xF8
};

uint8_t appEui[] = {
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00
};

uint8_t appKey[] = {
  0x26, 0x8B, 0xCE, 0x31,
  0x4C, 0x8B, 0x02, 0x80,
  0x0E, 0x33, 0x6B, 0x59,
  0xFC, 0x2D, 0x43, 0x47
};

// =====================================================
// ABP PARAMETERS - NOT USED FOR OTAA
// =====================================================
uint8_t nwkSKey[] = {
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00
};

uint8_t appSKey[] = {
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00
};

uint32_t devAddr = 0x00000000;

// =====================================================
// AS923 CHANNEL MASK
// 0x0003 = 923.2 MHz + 923.4 MHz
// =====================================================
uint16_t userChannelsMask[6] = {
  0x0003, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000
};

// =====================================================
// LORAWAN SETTINGS
// =====================================================
LoRaMacRegion_t loraWanRegion = ACTIVE_REGION;
DeviceClass_t loraWanClass = CLASS_A;

bool overTheAirActivation = true;
bool loraWanAdr = false;
bool isTxConfirmed = false;

// keepNet true = after joined, do not keep rejoining every reset if session can be kept.
bool keepNet = true;

// Keep false while testing USB serial.
bool lowPowerEn = false;

uint8_t appPort = 2;
uint8_t confirmedNbTrials = 1;

// =====================================================
// SEND INTERVAL
// =====================================================
uint32_t appTxDutyCycle = 30000;

// Keep counter across soft reset / sleep wake.
RTC_DATA_ATTR uint16_t payloadCounter = 0;
RTC_DATA_ATTR uint32_t bootCount = 0;

// =====================================================
// SENSOR PINS - Heltec WiFi LoRa 32 V4 ESP32-S3
// =====================================================
#define I2C_SDA 47
#define I2C_SCL 48
#define MPU_ADDR 0x68
#define ONE_WIRE_BUS 7

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature ds18b20(&oneWire);

// =====================================================
// SENSOR VARIABLES
// =====================================================
bool mpu_ok = false;
bool ds_ok = false;

float ax_g = 0.0;
float ay_g = 0.0;
float az_g = 0.0;

float gx_dps = 0.0;
float gy_dps = 0.0;
float gz_dps = 0.0;

float mpu_temp_c = 0.0;
float ds18b20_c = -127.0;

// =====================================================
// RESET REASON DEBUG
// =====================================================
void printResetReason() {
  esp_reset_reason_t reason = esp_reset_reason();

  Serial.print("[RESET REASON] ");

  switch (reason) {
    case ESP_RST_POWERON:
      Serial.println("POWERON reset");
      break;
    case ESP_RST_EXT:
      Serial.println("External reset");
      break;
    case ESP_RST_SW:
      Serial.println("Software reset");
      break;
    case ESP_RST_PANIC:
      Serial.println("Panic / crash reset");
      break;
    case ESP_RST_INT_WDT:
      Serial.println("Interrupt watchdog reset");
      break;
    case ESP_RST_TASK_WDT:
      Serial.println("Task watchdog reset");
      break;
    case ESP_RST_WDT:
      Serial.println("Other watchdog reset");
      break;
    case ESP_RST_DEEPSLEEP:
      Serial.println("Deep sleep wakeup");
      break;
    case ESP_RST_BROWNOUT:
      Serial.println("Brownout reset - POWER PROBLEM");
      break;
    case ESP_RST_SDIO:
      Serial.println("SDIO reset");
      break;
    default:
      Serial.println("Unknown reset reason");
      break;
  }
}

// =====================================================
// MPU6050 FUNCTIONS
// =====================================================
bool mpuWriteReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool mpuReadBytes(uint8_t reg, uint8_t *buffer, uint8_t length) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  uint8_t received = Wire.requestFrom(MPU_ADDR, length);

  if (received != length) {
    return false;
  }

  for (int i = 0; i < length; i++) {
    buffer[i] = Wire.read();
  }

  return true;
}

bool initMPU6050() {
  Serial.println("[MPU6050] Initializing...");

  Wire.beginTransmission(MPU_ADDR);

  if (Wire.endTransmission() != 0) {
    Serial.println("[MPU6050] NOT FOUND at 0x68");
    return false;
  }

  // Wake up MPU6050.
  if (!mpuWriteReg(0x6B, 0x00)) {
    Serial.println("[MPU6050] Wake up failed");
    return false;
  }

  delay(100);

  // Accelerometer +/-2g.
  mpuWriteReg(0x1C, 0x00);

  // Gyroscope +/-250 dps.
  mpuWriteReg(0x1B, 0x00);

  Serial.println("[MPU6050] OK");
  return true;
}

bool readMPU6050() {
  uint8_t raw[14];

  if (!mpuReadBytes(0x3B, raw, 14)) {
    mpu_ok = false;
    return false;
  }

  int16_t accX = (int16_t)((raw[0] << 8) | raw[1]);
  int16_t accY = (int16_t)((raw[2] << 8) | raw[3]);
  int16_t accZ = (int16_t)((raw[4] << 8) | raw[5]);

  int16_t tempRaw = (int16_t)((raw[6] << 8) | raw[7]);

  int16_t gyroX = (int16_t)((raw[8] << 8) | raw[9]);
  int16_t gyroY = (int16_t)((raw[10] << 8) | raw[11]);
  int16_t gyroZ = (int16_t)((raw[12] << 8) | raw[13]);

  ax_g = accX / 16384.0;
  ay_g = accY / 16384.0;
  az_g = accZ / 16384.0;

  gx_dps = gyroX / 131.0;
  gy_dps = gyroY / 131.0;
  gz_dps = gyroZ / 131.0;

  mpu_temp_c = (tempRaw / 340.0) + 36.53;

  mpu_ok = true;
  return true;
}

bool readDS18B20() {
  ds18b20.requestTemperatures();

  float temp = ds18b20.getTempCByIndex(0);

  if (temp == DEVICE_DISCONNECTED_C || temp < -50 || temp > 100) {
    ds_ok = false;
    ds18b20_c = -127.0;
    return false;
  }

  ds_ok = true;
  ds18b20_c = temp;
  return true;
}

void readSensors() {
  mpu_ok = false;
  ds_ok = false;

  readMPU6050();
  readDS18B20();

#if DEBUG_SENSOR_PRINT
  Serial.println();
  Serial.println("========== SENSOR DATA ==========");

  Serial.print("MPU6050 OK: ");
  Serial.println(mpu_ok ? "YES" : "NO");

  Serial.print("DS18B20 OK: ");
  Serial.println(ds_ok ? "YES" : "NO");

  Serial.print("AX(g): ");
  Serial.print(ax_g, 3);
  Serial.print(" | AY(g): ");
  Serial.print(ay_g, 3);
  Serial.print(" | AZ(g): ");
  Serial.println(az_g, 3);

  Serial.print("GX(dps): ");
  Serial.print(gx_dps, 3);
  Serial.print(" | GY(dps): ");
  Serial.print(gy_dps, 3);
  Serial.print(" | GZ(dps): ");
  Serial.println(gz_dps, 3);

  Serial.print("MPU Temp(C): ");
  Serial.println(mpu_temp_c, 2);

  Serial.print("DS18B20 Temp(C): ");
  Serial.println(ds18b20_c, 2);

  Serial.println("=================================");
#endif
}

// =====================================================
// PAYLOAD HELPER
// =====================================================
void putInt16BE(uint8_t index, int16_t value) {
  appData[index] = highByte(value);
  appData[index + 1] = lowByte(value);
}

// =====================================================
// PREPARE 20 BYTES PAYLOAD
// =====================================================
static void prepareTxFrame(uint8_t port) {
  readSensors();

  appDataSize = 20;

  uint8_t status = 0;

  if (mpu_ok) {
    status |= 0x01;
  }

  if (ds_ok) {
    status |= 0x02;
  }

  // Accelerometer: g -> mg
  int16_t ax_mg = (int16_t)(ax_g * 1000.0);
  int16_t ay_mg = (int16_t)(ay_g * 1000.0);
  int16_t az_mg = (int16_t)(az_g * 1000.0);

  // Gyroscope: dps -> centi-dps
  int16_t gx_cdps = (int16_t)(gx_dps * 100.0);
  int16_t gy_cdps = (int16_t)(gy_dps * 100.0);
  int16_t gz_cdps = (int16_t)(gz_dps * 100.0);

  // Temperature: C -> centi-C
  int16_t mpu_temp_cc = (int16_t)(mpu_temp_c * 100.0);
  int16_t ds_temp_cc = (int16_t)(ds18b20_c * 100.0);

  putInt16BE(0, payloadCounter);

  appData[2] = status;
  appData[3] = 0x00;

  putInt16BE(4, ax_mg);
  putInt16BE(6, ay_mg);
  putInt16BE(8, az_mg);

  putInt16BE(10, gx_cdps);
  putInt16BE(12, gy_cdps);
  putInt16BE(14, gz_cdps);

  putInt16BE(16, mpu_temp_cc);
  putInt16BE(18, ds_temp_cc);

#if DEBUG_PAYLOAD_PRINT
  Serial.println();
  Serial.println("========== PREPARE SENSOR UPLINK ==========");

  Serial.print("Payload Counter: ");
  Serial.println(payloadCounter);

  Serial.print("Payload Size: ");
  Serial.print(appDataSize);
  Serial.println(" bytes");

  Serial.print("Status Byte: 0x");
  if (status < 0x10) {
    Serial.print("0");
  }
  Serial.println(status, HEX);

  Serial.print("AX(g): ");
  Serial.print(ax_g, 3);
  Serial.print(" | AY(g): ");
  Serial.print(ay_g, 3);
  Serial.print(" | AZ(g): ");
  Serial.println(az_g, 3);

  Serial.print("GX(dps): ");
  Serial.print(gx_dps, 3);
  Serial.print(" | GY(dps): ");
  Serial.print(gy_dps, 3);
  Serial.print(" | GZ(dps): ");
  Serial.println(gz_dps, 3);

  Serial.print("MPU Temp(C): ");
  Serial.print(mpu_temp_c, 2);
  Serial.print(" | DS18B20 Temp(C): ");
  Serial.println(ds18b20_c, 2);

  Serial.print("Payload HEX: ");
  for (int i = 0; i < appDataSize; i++) {
    if (appData[i] < 0x10) {
      Serial.print("0");
    }
    Serial.print(appData[i], HEX);
    Serial.print(" ");
  }
  Serial.println();

  Serial.println("========== PREPARE DONE ==========");
#endif

  payloadCounter++;
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1500);

  bootCount++;

  Serial.println();
  Serial.println("=================================================");
  Serial.println(" HELTEC WiFi LoRa 32 V4 - OTAA SENSOR TX");
  Serial.println(" Region: AS923");
  Serial.println(" Activation: OTAA");
  Serial.println(" Class: A");
  Serial.println(" Payload: 20 bytes");
  Serial.println(" Interval: 30 seconds");
  Serial.println(" Data Rate: DR3");
  Serial.println(" Includes: AX AY AZ GX GY GZ + temperature");
  Serial.println("=================================================");

  Serial.print("[BOOT COUNT] ");
  Serial.println(bootCount);

  Serial.print("[CURRENT PAYLOAD COUNTER] ");
  Serial.println(payloadCounter);

  printResetReason();

  Mcu.begin(HELTEC_BOARD, SLOW_CLK_TPYE);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);

  ds18b20.begin();

  bool mpuInitResult = initMPU6050();

  if (mpuInitResult) {
    Serial.println("[SETUP] MPU6050 ready");
  } else {
    Serial.println("[SETUP] MPU6050 not ready, but LoRaWAN will still run");
  }

  Serial.println("[SETUP] Setup done");
  Serial.flush();
}

// =====================================================
// LOOP - HELTEC STATE MACHINE
// =====================================================
void loop() {
  switch (deviceState) {

    case DEVICE_STATE_INIT:
    {
#if (LORAWAN_DEVEUI_AUTO)
      LoRaWAN.generateDeveuiByChipID();
#endif

      Serial.println();
      Serial.println("[STATE] INIT");

      LoRaWAN.init(loraWanClass, loraWanRegion);

      // DR3 or higher is required for this 20-byte payload.
      LoRaWAN.setDefaultDR(3);

      Serial.println("[STATE] INIT DONE -> JOIN");
      Serial.flush();

      deviceState = DEVICE_STATE_JOIN;
      break;
    }

    case DEVICE_STATE_JOIN:
    {
      Serial.println();
      Serial.println("[STATE] JOINING...");
      Serial.flush();

      LoRaWAN.join();
      break;
    }

    case DEVICE_STATE_SEND:
    {
      Serial.println();
      Serial.println("[STATE] SEND");

      // Force DR3 before every uplink.
      LoRaWAN.setDefaultDR(3);

      prepareTxFrame(appPort);

      // Important: finish USB serial before radio TX.
      Serial.flush();
      delay(50);

#if DEBUG_SEND_PRINT
      Serial.println("[SEND] Calling LoRaWAN.send()");
      Serial.flush();
#endif

      LoRaWAN.send();

      // Do not print after LoRaWAN.send().
      deviceState = DEVICE_STATE_CYCLE;
      break;
    }

    case DEVICE_STATE_CYCLE:
    {
      Serial.println();
      Serial.println("[STATE] CYCLE");

      txDutyCycleTime = appTxDutyCycle;

      Serial.print("[INFO] Next uplink after ");
      Serial.print(txDutyCycleTime / 1000);
      Serial.println(" seconds");

      Serial.flush();

      LoRaWAN.cycle(txDutyCycleTime);
      deviceState = DEVICE_STATE_SLEEP;
      break;
    }

    case DEVICE_STATE_SLEEP:
    {
      LoRaWAN.sleep(loraWanClass);
      break;
    }

    default:
    {
      Serial.println("[STATE] UNKNOWN -> INIT");
      Serial.flush();

      deviceState = DEVICE_STATE_INIT;
      break;
    }
  }
}