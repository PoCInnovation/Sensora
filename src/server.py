import asyncio
import os
import json
import numpy as np
from dotenv import load_dotenv
from bleak import BleakClient

# --- CONFIGURATION ---
load_dotenv()
ADDRESS_MAC = os.getenv("ESP32_MAC_ADRESS")
CHARACTERISTIC_UUID = "3c5454f6-b1f7-4206-89f9-04677f4f467d"
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "5001"))

MAX_SERVOS = 16
MULTIPLIER_ANGLE = 90  # normalized 0-1 → 0-90°


async def send_matrix(client, matrix):
    """Send each cell of the matrix as a [angle, servo_id] BLE write."""
    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            servo_id = r * cols + c
            if servo_id >= MAX_SERVOS:
                continue
            angle = int(np.clip(matrix[r, c] * MULTIPLIER_ANGLE, 0, 180))
            await client.write_gatt_char(CHARACTERISTIC_UUID, bytearray([angle, servo_id]))


class MatrixServer:
    """TCP server that receives depth matrices from the IA pipeline and
    forwards them to the Arduino via BLE."""

    def __init__(self):
        self._ble_client = None
        self._ble_lock = asyncio.Lock()

    async def _ensure_ble_connected(self):
        if self._ble_client and self._ble_client.is_connected:
            return True
        try:
            print(f"[*] Connecting to BLE device {ADDRESS_MAC}...")
            self._ble_client = BleakClient(ADDRESS_MAC)
            await self._ble_client.connect()
            print("[+] BLE connected.")
            return True
        except Exception as e:
            print(f"[-] BLE connection failed: {e}")
            self._ble_client = None
            return False

    async def _handle_pipeline(self, reader, writer):
        addr = writer.get_extra_info("peername")
        print(f"[+] Pipeline connected: {addr}")
        buffer = ""

        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8")

                # Protocol: newline-delimited JSON — each line is a 2D matrix
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    try:
                        matrix = np.array(json.loads(line), dtype=float)
                        async with self._ble_lock:
                            if await self._ensure_ble_connected():
                                await send_matrix(self._ble_client, matrix)
                                response = {"status": "ok"}
                            else:
                                response = {"status": "error", "message": "BLE not connected"}
                        writer.write((json.dumps(response) + "\n").encode())
                        await writer.drain()
                    except (json.JSONDecodeError, ValueError) as e:
                        writer.write((json.dumps({"status": "error", "message": str(e)}) + "\n").encode())
                        await writer.drain()

        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            print(f"[-] Pipeline disconnected: {addr}")
            writer.close()

    async def start(self):
        if not ADDRESS_MAC:
            print("[!] Error: ESP32_MAC_ADDRESS is missing from .env")
            return

        server = await asyncio.start_server(self._handle_pipeline, SERVER_HOST, SERVER_PORT)
        print(f"[*] Matrix server listening on {SERVER_HOST}:{SERVER_PORT}")
        print(f"[*] BLE target: {ADDRESS_MAC}")

        async with server:
            await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(MatrixServer().start())
