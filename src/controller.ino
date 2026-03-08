#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

#define SERVICE_UUID        "12345678-1234-1234-1234-1234567890ab"
#define CHARACTERISTIC_UUID "abcd1234-5678-90ab-cdef-1234567890ab"

#define SERVO_MIN 150
#define SERVO_MAX 600
#define NB_SERVOS 16

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
BLECharacteristic *pCharacteristic;

uint16_t angleToPulse(int angle) {
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

void moveServo(int ch, int angle) {
  if (ch < 0 || ch >= NB_SERVOS) return;
  if (angle < 0 || angle > 180) return;
  pwm.setPWM(ch, 0, angleToPulse(angle));
}

void moveMultipleServos(String list, int angle) {
  list.trim();

  if (list == "ALL") {
    for (int ch = 0; ch < NB_SERVOS; ch++) {
      moveServo(ch, angle);
    }
    return;
  }

  int start = 0;
  while (true) {
    int comma = list.indexOf(',', start);
    String token = (comma == -1)
                   ? list.substring(start)
                   : list.substring(start, comma);

    moveServo(token.toInt(), angle);

    if (comma == -1) break;
    start = comma + 1;
  }
}

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pChar) override {
    std::string value = pChar->getValue();
    if (value.length() == 0) return;

    String cmd = String(value.c_str());
    Serial.print("Commande reçue : ");
    Serial.println(cmd);

    int colon = cmd.indexOf(':');
    if (colon == -1) return;
    if (cmd[0] != 'S') return;

    String servoPart = cmd.substring(1, colon);
    int angle = cmd.substring(colon + 1).toInt();

    moveMultipleServos(servoPart, angle);
  }
};

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("BOOT OK");

  Wire.begin(A4, A5);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);
  Serial.println("PCA9685 prêt");

  BLEDevice::init("NanoESP32-Servos");
  BLEServer *pServer = BLEDevice::createServer();
  BLEService *pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );

  pCharacteristic->setCallbacks(new MyCallbacks());
  pService->start();

  BLEDevice::getAdvertising()->start();
  Serial.println("BLE prêt pour commandes servos");
}

void loop() {
}
