#!/usr/bin/env python3
"""
PC Client for Servo Control
Sends commands to the Raspberry Pi servo server.

Usage:
    python3 servo_client.py

Or import and use programmatically:
    from servo_client import ServoClient
    client = ServoClient("192.168.1.100")
    client.set(0, 90)
"""

import socket
import json


class ServoClient:
    def __init__(self, host, port=5000):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        """Connect to the servo server."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"Connected to {self.host}:{self.port}")

    def disconnect(self):
        """Disconnect from the server."""
        if self.sock:
            self.sock.close()
            self.sock = None

    def _send(self, cmd_dict):
        """Send a command and receive response."""
        if not self.sock:
            self.connect()

        msg = json.dumps(cmd_dict) + "\n"
        self.sock.sendall(msg.encode("utf-8"))

        # Receive response
        response = ""
        while "\n" not in response:
            response += self.sock.recv(4096).decode("utf-8")

        return json.loads(response.strip())

    def ping(self):
        """Test connection."""
        return self._send({"cmd": "ping"})

    def set(self, servo, angle):
        """Set a single servo angle."""
        return self._send({"cmd": "set", "servo": servo, "angle": angle})

    def set_multiple(self, angles_dict):
        """
        Set multiple servos at once.
        angles_dict: {servo_id: angle, ...}
        """
        return self._send({"cmd": "set_multiple", "angles": angles_dict})

    def set_all(self, angle):
        """Set all servos to the same angle."""
        return self._send({"cmd": "set_all", "angle": angle})

    def center(self):
        """Center all servos to 90 degrees."""
        return self._send({"cmd": "center"})

    def release(self, servo):
        """Release a single servo."""
        return self._send({"cmd": "release", "servo": servo})

    def release_all(self):
        """Release all servos."""
        return self._send({"cmd": "release_all"})

    def get(self, servo):
        """Get current angle of a servo."""
        return self._send({"cmd": "get", "servo": servo})

    def sweep(self, servo, start=0, end=180, step=5, delay=0.05):
        """Sweep a servo."""
        return self._send({
            "cmd": "sweep",
            "servo": servo,
            "start": start,
            "end": end,
            "step": step,
            "delay": delay
        })

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# Interactive mode
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 servo_client.py <pi_ip_address>")
        print("Example: python3 servo_client.py 192.168.1.100")
        sys.exit(1)

    host = sys.argv[1]

    with ServoClient(host) as client:
        print("Servo Client - Interactive Mode")
        print("Commands:")
        print("  set <servo> <angle>    - Set servo angle")
        print("  all <angle>            - Set all servos")
        print("  center                 - Center all servos")
        print("  sweep <servo>          - Sweep a servo")
        print("  release                - Release all servos")
        print("  quit                   - Exit")
        print()

        # Test connection
        result = client.ping()
        print(f"Connection test: {result}")
        print()

        while True:
            try:
                cmd = input("> ").strip().split()
                if not cmd:
                    continue

                if cmd[0] == "quit":
                    break
                elif cmd[0] == "set" and len(cmd) == 3:
                    result = client.set(int(cmd[1]), float(cmd[2]))
                    print(result)
                elif cmd[0] == "all" and len(cmd) == 2:
                    result = client.set_all(float(cmd[1]))
                    print(result)
                elif cmd[0] == "center":
                    result = client.center()
                    print(result)
                elif cmd[0] == "sweep" and len(cmd) >= 2:
                    result = client.sweep(int(cmd[1]))
                    print(result)
                elif cmd[0] == "release":
                    result = client.release_all()
                    print(result)
                else:
                    print("Unknown command")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("Goodbye!")
