import asyncio
from bleak import BleakClient, BleakScanner
import os
from dotenv import load_dotenv

load_dotenv()
ADDRESS_MAC = os.getenv("ESP32_MAC_ADRESS")
UUID_CHARACTERISTIC = "3c5454f6-b1f7-4206-89f9-04677f4f467d"

async def scan_devices():
    print("Scanning for Bluetooth devices...")
    devices = await BleakScanner.discover()
    if devices:
        print("\nAppareils trouvés:")
        for device in devices:
            print(f"  - {device.address} ({device.name})")
    else:
        print("Aucun appareil trouvé")
    return devices

async def send_matrix():
    if not ADDRESS_MAC:
        raise ValueError("ESP32_MAC_ADRESS environment variable is not set")

    print(f"Connexion attempt to {ADDRESS_MAC}...")

    async with BleakClient(ADDRESS_MAC) as client:
        print("Connected with ESP32")

        matrix = bytearray([90, 0, 255])
        await client.write_gatt_char(UUID_CHARACTERISTIC, matrix)
        print("Matrix send")

async def main():
    await scan_devices()
    print("\n" + "="*50 + "\n")
    await send_matrix()

asyncio.run(main())

#UID pour le ESP32 -> f46d35c6-518c-44d4-8fe4-bba375eea5a9