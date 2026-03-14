import asyncio
import os
import numpy as np
from dotenv import load_dotenv
from bleak import BleakClient

# --- CONFIGURATION ---
load_dotenv()
ADDRESS_MAC = os.getenv("ESP32_MAC_ADRESS")
CHARACTERISTIC_UUID = "3c5454f6-b1f7-4206-89f9-04677f4f467d"

# -- PARAMETERS
MAX_SERVOS = 16
ANGLE_MAX_HARDWARE = 180
MULTIPLIER_ANGLE = 90
SLEEP_BETWEEN_COMMANDS = 0.01

class ServoController:
    def __init__(self, client):
        self.client = client
        self.positions_history = {}

    def _convert_to_angle(self, normalized_value):
        angle = int(normalized_value * MULTIPLIER_ANGLE)
        return max(0, min(ANGLE_MAX_HARDWARE, angle))

    async def send_servo_command(self, servo_id, angle):
        payload = bytearray([angle, servo_id])
        try:
            await self.client.write_gatt_char(CHARACTERISTIC_UUID, payload)
            self.positions_history[servo_id] = angle
            await asyncio.sleep(SLEEP_BETWEEN_COMMANDS)
        except Exception as e:
            print(f"[-] Failed to send Servo {servo_id}: {e}")

    async def process_matrix(self, matrix):
        rows, cols = matrix.shape

        for r in range(rows):
            for c in range(cols):
                servo_id = r * cols + c

                if servo_id >= MAX_SERVOS:
                    continue

                target_angle = self._convert_to_angle(matrix[r, c])
                last_angle = self.positions_history.get(servo_id, -1)

                if target_angle != last_angle:
                    await self.send_servo_command(servo_id, target_angle)

async def run_sync_process(matrix):
    if not ADDRESS_MAC:
        print("[!] Error: ESP32_MAC_ADDRESS is missing from .env")
        return

    print(f"[*] Attempting to connect : {ADDRESS_MAC}...")

    try:
        async with BleakClient(ADDRESS_MAC) as client:
            print("[+] Bluetooth Connected.")
            controller = ServoController(client)
            await controller.process_matrix(matrix)
            print("[+] Synchronization complete.")
    except Exception as e:
        print(f"[!] Connection error : {e}")
