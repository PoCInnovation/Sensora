#!/usr/bin/env python3
"""
Network Server for Servo Control
Runs on the Raspberry Pi, accepts commands from PC via TCP.

Usage on Pi:
    python3 servo_server.py

Commands (JSON over TCP):
    {"cmd": "set", "servo": 0, "angle": 90}
    {"cmd": "set_multiple", "angles": {0: 45, 1: 90, 5: 135}}
    {"cmd": "set_all", "angle": 90}
    {"cmd": "center"}
    {"cmd": "release", "servo": 0}
    {"cmd": "release_all"}
    {"cmd": "get", "servo": 0}
    {"cmd": "sweep", "servo": 0, "start": 0, "end": 180}
"""

import socket
import json
import threading
from servo_controller import MultiServoController

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000


class ServoServer:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.controller = MultiServoController(num_servos=36)
        self.running = False

    def handle_command(self, data):
        """Process a command and return response."""
        try:
            cmd = data.get("cmd")

            if cmd == "set":
                servo = data["servo"]
                angle = data["angle"]
                self.controller.set_angle(servo, angle)
                return {"status": "ok", "servo": servo, "angle": angle}

            elif cmd == "set_multiple":
                angles = {int(k): v for k, v in data["angles"].items()}
                self.controller.set_angles(angles)
                return {"status": "ok", "updated": len(angles)}

            elif cmd == "set_all":
                angle = data["angle"]
                self.controller.set_all(angle)
                return {"status": "ok", "angle": angle}

            elif cmd == "center":
                self.controller.center_all()
                return {"status": "ok", "action": "centered"}

            elif cmd == "release":
                servo = data["servo"]
                self.controller.release(servo)
                return {"status": "ok", "servo": servo, "action": "released"}

            elif cmd == "release_all":
                self.controller.release_all()
                return {"status": "ok", "action": "all_released"}

            elif cmd == "get":
                servo = data["servo"]
                angle = self.controller.get_angle(servo)
                return {"status": "ok", "servo": servo, "angle": angle}

            elif cmd == "sweep":
                servo = data["servo"]
                start = data.get("start", 0)
                end = data.get("end", 180)
                step = data.get("step", 5)
                delay = data.get("delay", 0.05)
                self.controller.sweep(servo, start, end, step, delay)
                return {"status": "ok", "servo": servo, "action": "sweep_complete"}

            elif cmd == "ping":
                return {"status": "ok", "message": "pong"}

            else:
                return {"status": "error", "message": f"Unknown command: {cmd}"}

        except KeyError as e:
            return {"status": "error", "message": f"Missing parameter: {e}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def handle_client(self, conn, addr):
        """Handle a single client connection."""
        print(f"Client connected: {addr}")
        buffer = ""

        try:
            while self.running:
                data = conn.recv(4096).decode("utf-8")
                if not data:
                    break

                buffer += data

                # Process complete JSON messages (newline-delimited)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        try:
                            cmd_data = json.loads(line)
                            response = self.handle_command(cmd_data)
                        except json.JSONDecodeError:
                            response = {"status": "error", "message": "Invalid JSON"}

                        conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

        except ConnectionResetError:
            pass
        finally:
            print(f"Client disconnected: {addr}")
            conn.close()

    def start(self):
        """Start the server."""
        self.running = True

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(5)
            server.settimeout(1.0)

            print(f"Servo server listening on {self.host}:{self.port}")
            print("Commands: set, set_multiple, set_all, center, release, release_all, get, sweep, ping")

            try:
                while self.running:
                    try:
                        conn, addr = server.accept()
                        thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                        thread.daemon = True
                        thread.start()
                    except socket.timeout:
                        continue
            except KeyboardInterrupt:
                print("\nShutting down...")
                self.running = False
                self.controller.release_all()


if __name__ == "__main__":
    server = ServoServer()
    server.start()
