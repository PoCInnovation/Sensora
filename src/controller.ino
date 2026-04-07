#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

#define SERVICE_UUID        "f46d35c6-518c-44d4-8fe4-bba375eea5a9"
#define CHARACTERISTIC_UUID "3c5454f6-b1f7-4206-89f9-04677f4f467d"

#define SERVO_MIN 150
#define SERVO_MAX 600
#define NB_SERVOS 36

Adafruit_PWMServoDriver pwmA = Adafruit_PWMServoDriver(0x40);
Adafruit_PWMServoDriver pwmB = Adafruit_PWMServoDriver(0x41);
Adafruit_PWMServoDriver pwmC = Adafruit_PWMServoDriver(0x42);
bool deviceConnected = false;
bool oldDeviceConnected = false;

uint16_t angleToPulse(int angle) {
  return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

void moveServo(int channel, int angle) {
  if (channel < 0 || channel >= NB_SERVOS) {
    return;
  }

  uint16_t pulse = angleToPulse(angle);

  if (channel < 16) {
    pwmA.setPWM(channel, 0, pulse);
  } else if (channel < 32) {
    pwmB.setPWM(channel - 16, 0, pulse);
  } else {
    pwmC.setPWM(channel - 32, 0, pulse);
  }
}

void applyAngleToServos(const uint8_t* data, size_t length) {
  if (length < 2 || data == NULL) {
    return;
  }

  int angle = data[0];
  if (angle > 180) {
    angle = 180;
  }

  Serial.print("[Action] Angle : ");
  Serial.print(angle);
  Serial.print("° on Servo : ");

  for (size_t index = 1; index < length; index++) {
    uint8_t servoNumber = data[index];

    if (servoNumber == 255) {
      Serial.print("ALL ");
      for (int channel = 0; channel < NB_SERVOS; channel++) {
        moveServo(channel, angle);
      }
    } else if (servoNumber < NB_SERVOS) {
      Serial.print(servoNumber);
      Serial.print(" ");
      moveServo(servoNumber, angle);
    }
  }

  Serial.println();
}

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) {
    (void)server;
    deviceConnected = true;
    Serial.println("\n>>> Device CONNECTED !");
  }

  void onDisconnect(BLEServer* server) {
    (void)server;
    deviceConnected = false;
    Serial.println("\n<<< Device DISCONNECTED.");
  }
};

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* characteristic) {
    size_t length = characteristic->getLength();
    uint8_t* data = characteristic->getData();
    applyAngleToServos(data, length);
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
  pwmC.begin();
  pwmC.setPWMFreq(50);
  Serial.println("1. PCA9685 #1 (0x40): OK (Frequence 50Hz)");
  Serial.println("2. PCA9685 #2 (0x41): OK (Frequence 50Hz)");
  Serial.println("3. PCA9685 #3 (0x42): OK (Frequence 50Hz)");

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

  Serial.println("4. Bluetooth : OK ('Sensora Device')");
  Serial.println("5. Status : Waiting for connexion...");
  Serial.println("--------------------------------------------------");
}

void loop() {
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    BLEDevice::getAdvertising()->start();
    Serial.println("... Advertising restarted (Visible) ...");
    oldDeviceConnected = deviceConnected;
  }

  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }
}