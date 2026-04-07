#!/usr/bin/env python3
"""
Servo Controller via BLE for ESP32
Controls 32 servos through Bluetooth Low Energy connection to ESP32

Hardware setup:
- ESP32 with BLE characteristic: 3c5454f6-b1f7-4206-89f9-04677f4f467d
- Depth matrix from vision system converted to servo angles
"""

import asyncio
import os
import numpy as np
from dotenv import load_dotenv
from bleak import BleakClient

# --- CONFIGURATION ---
load_dotenv()
ADDRESS_MAC = os.getenv("ESP32_MAC_ADRESS") or os.getenv("ESP32_MAC_ADDRESS")
CHARACTERISTIC_UUID = "3c5454f6-b1f7-4206-89f9-04677f4f467d"

# -- PARAMETERS
MAX_SERVOS = 36
ANGLE_MAX_HARDWARE = 180
MULTIPLIER_ANGLE = 90
SLEEP_BETWEEN_COMMANDS = 0.05
SMOOTH_STEP_SIZE = 5
SMOOTH_STEP_DELAY = 0.02
DEPTH_PATCHES = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
])


class BleServoController:
    """Control servos via BLE connection to ESP32."""

    def __init__(self, client):
        self.client = client
        self.positions_history = {}

    def _convert_to_angle(self, normalized_value):
        """Convert normalized depth value to servo angle."""
        angle = int(normalized_value * MULTIPLIER_ANGLE)
        return max(0, min(ANGLE_MAX_HARDWARE, angle))

    async def send_servo_command(self, servo_id, angle):
        """Send servo command via BLE."""
        payload = bytearray([angle, servo_id])
        try:
            await self.client.write_gatt_char(CHARACTERISTIC_UUID, payload)
            self.positions_history[servo_id] = angle
            await asyncio.sleep(SLEEP_BETWEEN_COMMANDS)
        except Exception as e:
            print(f"[-] Failed to send Servo {servo_id}: {e}")

    async def move_smooth(self, servo_id, current_angle, target_angle):
        """Move servo smoothly from current to target angle."""
        if current_angle == -1:
            await self.send_servo_command(servo_id, target_angle)
            return

        delta = target_angle - current_angle
        steps = abs(delta) // SMOOTH_STEP_SIZE
        direction = 1 if delta > 0 else -1

        for i in range(1, steps + 1):
            intermediate = current_angle + direction * SMOOTH_STEP_SIZE * i
            await self.send_servo_command(servo_id, intermediate)
            await asyncio.sleep(SMOOTH_STEP_DELAY)

        if self.positions_history.get(servo_id) != target_angle:
            await self.send_servo_command(servo_id, target_angle)

    async def process_matrix(self, matrix):
        """Process depth matrix and update servo positions."""
        rows, cols = matrix.shape

        for r in range(rows):
            for c in range(cols):
                servo_id = r * cols + c

                if servo_id >= MAX_SERVOS:
                    continue

                target_angle = self._convert_to_angle(matrix[r, c])
                last_angle = self.positions_history.get(servo_id, -1)

                if target_angle != last_angle:
                    await self.move_smooth(servo_id, last_angle, target_angle)


async def run_sync_process(matrix):
    """Connect to ESP32 via BLE and synchronize servo positions."""
    if not ADDRESS_MAC:
        print("[!] Error: ESP32_MAC_ADDRESS is missing from .env")
        return

    print(f"[*] Attempting to connect : {ADDRESS_MAC}...")

    try:
        async with BleakClient(ADDRESS_MAC) as client:
            print("[+] Bluetooth Connected.")
            controller = BleServoController(client)
            await controller.process_matrix(matrix)
            print("[+] Synchronization complete.")
    except Exception as e:
        print(f"[!] Connection error : {e}")


if __name__ == "__main__":
    # Launch asyncio loop with test depth matrix
    asyncio.run(run_sync_process(DEPTH_PATCHES))
