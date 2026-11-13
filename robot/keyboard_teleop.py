"""
Drive the robot from a keyboard, using exactly the same messages as the EEG
decoder. Used to verify the Pi link and tune the gait independently of EEG.

  python robot/keyboard_teleop.py --host 192.168.1.50     # the Pi's IP

  W / Up     FORWARD        A / Left   LEFT
  S / Down   BACKWARD       D / Right  RIGHT
  Space      STOP           Q          quit
Commands are re-sent at 10 Hz to keep the receiver watchdog satisfied.
"""
import argparse
import curses
import json
import socket
import time

KEYS = {ord("w"): "FORWARD", curses.KEY_UP: "FORWARD", ord("s"): "BACKWARD", curses.KEY_DOWN: "BACKWARD",
        ord("a"): "LEFT", curses.KEY_LEFT: "LEFT", ord("d"): "RIGHT", curses.KEY_RIGHT: "RIGHT",
        ord(" "): "STOP"}


def run(scr, host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    curses.curs_set(0)
    scr.nodelay(True)
    cmd, seq = "STOP", 0
    while True:
        k = scr.getch()
        if k in (ord("q"), ord("Q")):
            cmd = "STOP"
        elif k in KEYS:
            cmd = KEYS[k]
        seq += 1
        sock.sendto(json.dumps({"command": cmd, "confidence": 1.0, "seq": seq, "t": time.time()}).encode(),
                    (host, port))
        scr.erase()
        scr.addstr(0, 0, f"sending to {host}:{port}   W/A/S/D or arrows, space = stop, q = quit")
        scr.addstr(2, 0, f"command: {cmd:<9}")
        scr.refresh()
        if k in (ord("q"), ord("Q")):
            return
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=5005)
    a = ap.parse_args()
    curses.wrapper(run, a.host, a.port)


if __name__ == "__main__":
    main()
