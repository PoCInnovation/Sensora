#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

#define SERVICE_UUID        "f46d35c6-518c-44d4-8fe4-bba375eea5a9"
#define CHARACTERISTIC_UUID "3c5454f6-b1f7-4206-89f9-04677f4f467d"

#define SERVO_MIN 150
#define SERVO_MAX 600
#define NB_SERVOS 32

Adafruit_PWMServoDriver pwmA = Adafruit_PWMServoDriver(0x40);
Adafruit_PWMServoDriver pwmB = Adafruit_PWMServoDriver(0x41);
bool deviceConnected = false;
bool oldDeviceConnected = false;

uint16_t angleToPulse(int angle) {
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

void moveServo(int ch, int angle) {
  if (ch < 0 || ch >= NB_SERVOS) return;
  uint16_t pulse = angleToPulse(angle);

  if (ch < 16) {
    pwmA.setPWM(ch, 0, pulse);
  } else {
    pwmB.setPWM(ch - 16, 0, pulse);
  }
}

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("\n>>> Device CONNECTED !");
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("\n<<< Device DISCONNECTED.");
    }
};

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pChar) override {
    std::string value = pChar->getValue();
    if (value.length() < 2) return;

    int angle = (uint8_t)value[0];
    if (angle > 180) angle = 180;

    Serial.print("[Action] Angle : ");
    Serial.print(angle);
    Serial.print("° on Servo : ");

    for (int i = 1; i < value.length(); i++) {
      uint8_t servoNum = (uint8_t)value[i];
      if (servoNum == 255) {
        Serial.print("ALL ");
        for (int ch = 0; ch < NB_SERVOS; ch++) moveServo(ch, angle);
      } else if (servoNum < NB_SERVOS) {
        Serial.print(servoNum);
        Serial.print(" ");
        moveServo(servoNum, angle);
      }
    }
    Serial.println();
  }
};

void setup() {
  Serial.begin(115200);

  Serial.println("\n--------------------------------------------------");
  Serial.println("--- SYSTEM INITIALISATION ---");

  Wire.begin();
  pwmA.begin();
  pwmA.setPWMFreq(50);
  pwmB.begin();
  pwmB.setPWMFreq(50);
  Serial.println("1. PCA9685 #1 (0x40): OK (Frequence 50Hz)");
  Serial.println("2. PCA9685 #2 (0x41): OK (Frequence 50Hz)");

  BLEDevice::init("Sensora Device");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  BLECharacteristic *pChar = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );

  pChar->setCallbacks(new MyCallbacks());
  pService->start();

  BLEDevice::getAdvertising()->start();
  
  Serial.println("3. Bluetooth : OK ('Sensora Device')");
  Serial.println("4. Status : Waiting for connexion...");
  Serial.println("--------------------------------------------------");
}

void loop() {
    if (!deviceConnected && oldDeviceConnected) {
        delay(500);
        BLEDevice::getAdvertising()->start();
        Serial.println("... Marketing relaunched (Visible) ...");
        oldDeviceConnected = deviceConnected;
    }
    
    if (deviceConnected && !oldDeviceConnected) {
        oldDeviceConnected = deviceConnected;
    }
}