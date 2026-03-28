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

# Calibration constants (override at call-site if needed)
HOME_ANGLE  = 0      # degrees — physical rest/home position
MIN_ANGLE   = 0      # hard lower bound for all servos
MAX_ANGLE   = 180    # hard upper bound for all servos
CALIB_STEP  = 3      # degrees per step during homing sweep
CALIB_DELAY = 0.02   # seconds between steps (smooth, not violent)
SETTLE_TIME = 0.5    # seconds to wait after all servos reach home


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

        # Software-tracked positions. None = unknown (not yet calibrated).
        # Populated by calibrate(); required before move_by() can be called.
        self._current_positions: dict[int, float] = {i: None for i in range(self.num_servos)}

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

    def calibrate(
        self,
        home_angle: float = HOME_ANGLE,
        step: float = CALIB_STEP,
        delay: float = CALIB_DELAY,
        settle_time: float = SETTLE_TIME,
    ) -> None:
        """
        Home all servos to home_angle by sweeping from the opposite extreme.

        Sweeping from the far end guarantees every servo physically reaches
        home_angle regardless of its unknown starting position.
        Blocks until complete, then sets _current_positions for all servos.
        """
        print(f"[Calibrate] Homing {self.num_servos} servos to {home_angle}°...")

        # Sweep from the opposite extreme toward home_angle
        if home_angle <= 90:
            start_angle = MAX_ANGLE
            step_signed = -abs(int(step))
            stop = int(home_angle) - 1   # inclusive endpoint for range
        else:
            start_angle = MIN_ANGLE
            step_signed = abs(int(step))
            stop = int(home_angle) + 1

        for angle in range(int(start_angle), stop, step_signed):
            for sid in range(self.num_servos):
                self.set_angle(sid, angle)
            time.sleep(delay)

        # Final precise set to exact home_angle
        for sid in range(self.num_servos):
            self.set_angle(sid, home_angle)
        time.sleep(settle_time)

        for sid in range(self.num_servos):
            self._current_positions[sid] = float(home_angle)

        print(f"[Calibrate] Done. All {self.num_servos} servos at {home_angle}°.")

    def move_by(self, servo_id: int, delta: float) -> float:
        """
        Move servo_id by delta degrees relative to its tracked position.

        Raises RuntimeError if calibrate() has not been called yet.
        Returns the new absolute angle (clamped to [MIN_ANGLE, MAX_ANGLE]).
        """
        current = self._current_positions.get(servo_id)
        if current is None:
            raise RuntimeError(
                f"Servo {servo_id} has no known position. Run calibrate() first."
            )
        new_angle = max(MIN_ANGLE, min(MAX_ANGLE, current + delta))
        self.set_angle(servo_id, new_angle)
        self._current_positions[servo_id] = new_angle
        return new_angle

    def move_all_by(self, deltas: dict) -> dict:
        """
        Move multiple servos by their respective deltas.

        deltas: {servo_id: delta_degrees, ...}
        Returns {servo_id: new_absolute_angle, ...}.
        """
        new_positions = {}
        for servo_id, delta in deltas.items():
            new_positions[int(servo_id)] = self.move_by(int(servo_id), float(delta))
        return new_positions

    def get_positions(self) -> dict:
        """Return a copy of the current tracked positions for all servos."""
        return dict(self._current_positions)


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
