"""
Drive the robot from a keyboard, using exactly the same messages as the EEG
decoder. Used to verify the Pi link and tune the gait independently of EEG.

  python robot/keyboard_teleop.py                         # HIRO access point (192.168.4.1)
  python robot/keyboard_teleop.py --host 192.168.1.50     # robot on another network

  W / Up     FORWARD        A / Left   LEFT
  S / Down   BACKWARD       D / Right  RIGHT
  Space      STOP           Q          quit
Commands are re-sent at 10 Hz to keep the receiver watchdog satisfied.
"""
import argparse
import json
import socket
import sys
import time

try:
    import curses
except ImportError:  # Windows without the windows-curses package
    curses = None

LETTERS = {"w": "FORWARD", "s": "BACKWARD", "a": "LEFT", "d": "RIGHT", " ": "STOP"}


def send(sock, addr, cmd, seq):
    sock.sendto(json.dumps({"command": cmd, "confidence": 1.0, "seq": seq, "t": time.time()}).encode(), addr)


def run_windows(host, port):
    """Console fallback for Windows (uses msvcrt instead of curses)."""
    import msvcrt
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd, seq, last = "STOP", 0, None
    print(f"sending to {host}:{port}   W/A/S/D, space = stop, q = quit")
    while True:
        if msvcrt.kbhit():
            k = msvcrt.getwch().lower()
            if k == "q":
                send(sock, (host, port), "STOP", seq + 1)
                return
            cmd = LETTERS.get(k, cmd)
        seq += 1
        send(sock, (host, port), cmd, seq)
        if cmd != last:
            print(f"command: {cmd}")
            last = cmd
        time.sleep(0.1)


def run(scr, host, port):
    KEYS = {ord(k): v for k, v in LETTERS.items()}
    KEYS.update({curses.KEY_UP: "FORWARD", curses.KEY_DOWN: "BACKWARD",
                 curses.KEY_LEFT: "LEFT", curses.KEY_RIGHT: "RIGHT"})
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
    ap.add_argument("--host", default="192.168.4.1",
                    help="robot IP (default: HIRO's own access point)")
    ap.add_argument("--port", type=int, default=5005)
    a = ap.parse_args()
    if curses is not None:
        curses.wrapper(run, a.host, a.port)
    elif sys.platform == "win32":
        run_windows(a.host, a.port)
    else:
        sys.exit("curses is not available; install it or run on Windows")


if __name__ == "__main__":
    main()
