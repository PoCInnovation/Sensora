#!/usr/bin/env python3
"""
Servo Controller for 36 servos via 3x PCA9685 boards
Raspberry Pi 5 with I2C

Hardware setup:
- PCA9685 #1: I2C address 0x40 (default) - Servos 0-15
- PCA9685 #2: I2C address 0x41 (A0 bridged) - Servos 16-31
- PCA9685 #3: I2C address 0x42 (A1 bridged) - Servos 32-47 (using 32-35)

Wiring:
- Pi SDA (GPIO 2) --> All PCA9685 SDA pins (daisy-chained)
- Pi SCL (GPIO 3) --> All PCA9685 SCL pins (daisy-chained)
- Pi 3.3V --> All PCA9685 VCC pins
- Pi GND --> All PCA9685 GND pins
- External 5-6V PSU --> All PCA9685 V+ terminals (servo power)
- External PSU GND --> All PCA9685 GND (common ground!)
"""

import time
from adafruit_servokit import ServoKit


class MultiServoController:
    def __init__(self, num_servos=36):
        self.num_servos = num_servos

        # Initialize 3 PCA9685 boards at different I2C addresses
        self.boards = [
            ServoKit(channels=16, address=0x40),  # Servos 0-15
            ServoKit(channels=16, address=0x41),  # Servos 16-31
            ServoKit(channels=16, address=0x42),  # Servos 32-47
        ]

        # Set PWM frequency (50Hz is standard for servos)
        for board in self.boards:
            board.frequency = 50

        # Default pulse width range (adjust per servo specs)
        # Common values: 500-2500us or 1000-2000us
        self.set_pulse_range_all(500, 2500)

        print(f"Initialized {num_servos} servos on {len(self.boards)} PCA9685 boards")

    def _get_board_and_channel(self, servo_id):
        """Convert global servo ID (0-35) to board index and channel."""
        if servo_id < 0 or servo_id >= self.num_servos:
            raise ValueError(f"Servo ID must be 0-{self.num_servos - 1}")
        board_idx = servo_id // 16
        channel = servo_id % 16
        return board_idx, channel

    def set_pulse_range(self, servo_id, min_pulse=500, max_pulse=2500):
        """Set pulse width range for a specific servo."""
        board_idx, channel = self._get_board_and_channel(servo_id)
        self.boards[board_idx].servo[channel].set_pulse_width_range(min_pulse, max_pulse)

    def set_pulse_range_all(self, min_pulse=500, max_pulse=2500):
        """Set pulse width range for all servos."""
        for i in range(self.num_servos):
            self.set_pulse_range(i, min_pulse, max_pulse)

    def set_angle(self, servo_id, angle):
        """Set a single servo to a specific angle (0-180)."""
        board_idx, channel = self._get_board_and_channel(servo_id)
        angle = max(0, min(180, angle))  # Clamp to valid range
        self.boards[board_idx].servo[channel].angle = angle

    def set_angles(self, angles_dict):
        """
        Set multiple servos at once.
        angles_dict: {servo_id: angle, ...}
        """
        for servo_id, angle in angles_dict.items():
            self.set_angle(servo_id, angle)

    def set_all(self, angle):
        """Set all servos to the same angle."""
        for i in range(self.num_servos):
            self.set_angle(i, angle)

    def get_angle(self, servo_id):
        """Get current angle of a servo."""
        board_idx, channel = self._get_board_and_channel(servo_id)
        return self.boards[board_idx].servo[channel].angle

    def center_all(self):
        """Center all servos to 90 degrees."""
        self.set_all(90)

    def release(self, servo_id):
        """Release a servo (stop sending PWM signal)."""
        board_idx, channel = self._get_board_and_channel(servo_id)
        self.boards[board_idx].servo[channel].angle = None

    def release_all(self):
        """Release all servos."""
        for i in range(self.num_servos):
            self.release(i)

    def sweep(self, servo_id, start=0, end=180, step=5, delay=0.05):
        """Sweep a servo from start to end angle."""
        if start < end:
            angles = range(start, end + 1, step)
        else:
            angles = range(start, end - 1, -step)

        for angle in angles:
            self.set_angle(servo_id, angle)
            time.sleep(delay)


# Quick test
if __name__ == "__main__":
    controller = MultiServoController(num_servos=36)

    print("Centering all servos...")
    controller.center_all()
    time.sleep(1)

    print("Testing servo 0 sweep...")
    controller.sweep(0, 0, 180)
    controller.sweep(0, 180, 90)

    print("Done. Releasing all servos.")
    controller.release_all()
