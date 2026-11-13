"""
Where commands go. Replace or extend `send` to drive your robot / wheelchair /
game. UDP JSON is a simple bridge to anything (Arduino via a small relay, ROS,
Unity, a Python robot controller...).
"""
import json
import socket


class PrintSink:
    def __init__(self, only_changes=True):
        self.last, self.only_changes = None, only_changes

    def send(self, d):
        if not self.only_changes or d["command"] != self.last:
            print(f"{d['command']:<9} ({d['label']} {d['confidence']:.2f})", flush=True)
        self.last = d["command"]


class UDPSink:
    def __init__(self, host="127.0.0.1", port=5005):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, d):
        self.sock.sendto(json.dumps(d).encode(), self.addr)
