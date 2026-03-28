#!/usr/bin/env python3
"""
Example animations for 36-servo system.
"""

import time
import math
from servo_client import ServoClient

PI_HOST = "192.168.1.100"  # Change to Pi's IP


def wave_animation(client, cycles=3, speed=0.05):
    """Wave pattern across all servos."""
    print("Running wave animation...")

    for _ in range(cycles):
        for phase in range(0, 360, 10):
            angles = {}
            for servo in range(36):
                # Sine wave with phase offset per servo
                angle = 90 + 45 * math.sin(math.radians(phase + servo * 20))
                angles[servo] = angle
            client.set_multiple(angles)
            time.sleep(speed)


def sequential_sweep(client, delay=0.1):
    """Sweep each servo one at a time."""
    print("Running sequential sweep...")

    for servo in range(36):
        client.set(servo, 0)
        time.sleep(delay)
        client.set(servo, 180)
        time.sleep(delay)
        client.set(servo, 90)


def breathing(client, cycles=5, speed=0.02):
    """All servos move together in breathing pattern."""
    print("Running breathing animation...")

    for _ in range(cycles):
        # Inhale
        for angle in range(45, 136, 2):
            client.set_all(angle)
            time.sleep(speed)
        # Exhale
        for angle in range(135, 44, -2):
            client.set_all(angle)
            time.sleep(speed)


def random_dance(client, duration=10, speed=0.1):
    """Random movements."""
    import random
    print(f"Running random dance for {duration}s...")

    start = time.time()
    while time.time() - start < duration:
        angles = {i: random.randint(30, 150) for i in range(36)}
        client.set_multiple(angles)
        time.sleep(speed)


def row_wave(client, servos_per_row=6, cycles=3, speed=0.1):
    """
    Treat servos as a grid and do row-by-row wave.
    Assumes 36 servos = 6 rows x 6 columns.
    """
    print("Running row wave...")
    num_rows = 36 // servos_per_row

    for _ in range(cycles):
        for row in range(num_rows):
            # Move this row up
            start_servo = row * servos_per_row
            angles = {start_servo + i: 135 for i in range(servos_per_row)}
            client.set_multiple(angles)
            time.sleep(speed)

            # Move back to center
            angles = {start_servo + i: 90 for i in range(servos_per_row)}
            client.set_multiple(angles)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        PI_HOST = sys.argv[1]

    print(f"Connecting to {PI_HOST}...")

    with ServoClient(PI_HOST) as client:
        # Test connection
        print(client.ping())

        # Start from centered position
        print("Centering all servos...")
        client.center()
        time.sleep(1)

        # Run animations
        try:
            wave_animation(client, cycles=2)
            time.sleep(0.5)

            breathing(client, cycles=3)
            time.sleep(0.5)

            row_wave(client, cycles=2)
            time.sleep(0.5)

            sequential_sweep(client)

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            print("Centering and releasing...")
            client.center()
            time.sleep(0.5)
            client.release_all()
            print("Done!")
