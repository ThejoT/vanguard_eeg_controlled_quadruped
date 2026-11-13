"""
Where commands go. Replace or extend `send` to drive your robot / wheelchair /
game. UDP JSON is a simple bridge to anything (Arduino via a small relay, ROS,
Unity, a Python robot controller...).
"""
import json
import socket
import time


class PrintSink:
    def __init__(self, only_changes=True):
        self.last, self.only_changes = None, only_changes

    def send(self, d):
        if not self.only_changes or d["command"] != self.last:
            print(f"{d['command']:<9} ({d['label']} {d['confidence']:.2f})", flush=True)
        self.last = d["command"]


class UDPSink:
    """Sends each decision as one JSON datagram (see robot/pi_receiver.py).
    A message goes out every decoder step, which doubles as the heartbeat for
    the robot's safety watchdog."""

    def __init__(self, host="127.0.0.1", port=5005):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, d):
        self.seq += 1
        msg = {"command": d["command"], "confidence": d["confidence"], "label": d["label"],
               "seq": self.seq, "t": time.time()}
        self.sock.sendto(json.dumps(msg).encode(), self.addr)
