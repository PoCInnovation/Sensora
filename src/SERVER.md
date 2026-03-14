# ESP32 Bluetooth Servo Matrix Controller

This Python script synchronizes a numerical matrix (NumPy) with a physical grid of servomoteurs via **Bluetooth Low Energy (BLE)**. It maps normalized values (0.0 to 1.0) to physical angles and transmits them to an ESP32 equipped with a PCA9685 PWM driver.

## ⚙️ How It Works

### 1. Data Mapping & Logic

The script processes a 2D NumPy array where each cell represents a specific servo motor.

* **Index Calculation**: Servos are addressed sequentially. For a matrix cell at `[row, col]`, the Servo ID is calculated as: $ID = row \times total\_columns + column$.


* 
**Angle Conversion**: Normalized values (0.0 to 1.0) are multiplied by a `MULTIPLIER_ANGLE` (default: 90) to determine the target degrees.


* 
**Hardware Safety**: Angles are strictly clamped between 0° and 180° to prevent mechanical damage to the servos.



### 2. Communication Protocol

The script communicates with the ESP32 using the **GATT (Generic Attribute Profile)** protocol over BLE.

* 
**Packet Structure**: Commands are sent as a 2-byte `bytearray`:


* **Byte 0**: Target Angle ($0$ to $180$).
* **Byte 1**: Servo ID ($0$ to $15$).


* 
**Targeting All**: While not used by default in the matrix loop, sending `255` as the Servo ID triggers the "All Servos" mode on the firmware.



### 3. Efficiency & Fluidity Features

To ensure smooth movement and prevent Bluetooth congestion:

* **Differential Updates**: The `ServoController` maintains a `positions_history` dictionary. It only transmits a command if the new angle differs from the last sent position.
* 
**Command Throttling**: A micro-delay (`0.01s`) is inserted between each BLE write. This prevents the PCA9685 and the ESP32 serial buffer from being overwhelmed, reducing "jitter" or buzzing sounds.


* 
**Connection Management**: The script uses an asynchronous context manager (`BleakClient`) to ensure the connection is cleanly opened and closed.



## 🛠 Project Structure

| Component | Description |
| --- | --- |
| **`ServoController`** | Class managing state, angle conversion, and BLE transmissions. |
| **`process_matrix`** | Iterates through the NumPy array and filters out indices exceeding `MAX_SERVOS`. |
| **`.env` File** | Stores the `ESP32_MAC_ADRESS` to keep the code portable and secure. |

## 🚀 Setup

1. **Environment**: Ensure `bleak`, `numpy`, and `python-dotenv` are installed.
2. **Configuration**: Update your `.env` file with your ESP32's MAC address.
3. **Execution**: Run the script. It will scan, connect, and move the servos according to `TEST_DATA`.
