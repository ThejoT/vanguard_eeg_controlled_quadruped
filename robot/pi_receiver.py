"""
Runs ON THE RASPBERRY PI. Receives motion commands over Wi-Fi (UDP) from the
EEG computer (live.py --robot PI_IP) or the keyboard tester, and drives the robot.

  python3 robot/pi_receiver.py --driver dryrun          # test the link, no motors
  python3 robot/pi_receiver.py --driver gait            # servo trot gait (calibrate robot_config.json)
  python3 robot/pi_receiver.py --driver mycode:MyDriver # your existing walking code

Only the Python standard library is needed (plus your driver's libraries).

Safety:
  * watchdog: if no message arrives for --timeout seconds (link lost, laptop
    crashed, decoder stopped), the robot STOPs.
  * --min-confidence: ignore commands the decoder isn't sure about.
  * only messages from --allow IP (if set) are accepted.
  * Ctrl+C stops the robot before exiting.

Message format (JSON over UDP, one per datagram):
  {"command": "FORWARD", "confidence": 0.81, "seq": 123, "t": 1700000000.0}
"""
import argparse
import json
import socket
import time

from drivers import load_driver

VALID = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--driver", default="dryrun")
    ap.add_argument("--config", help="robot_config.json for the gait driver")
    ap.add_argument("--dry", action="store_true", help="gait driver: print servo angles, don't move")
    ap.add_argument("--timeout", type=float, default=1.0, help="seconds without a message before STOP")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--allow", help="only accept commands from this IP")
    ap.add_argument("--rate", type=float, default=50.0, help="driver update rate (Hz)")
    ap.add_argument("--max-runtime", type=float, default=0, help="exit after N seconds (testing)")
    a = ap.parse_args()

    driver = load_driver(a.driver, a.config, a.dry)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", a.port))
    sock.settimeout(1.0 / a.rate)
    print(f"listening on UDP :{a.port}  driver={a.driver}  watchdog={a.timeout}s", flush=True)

    driver.start()
    current, last_msg, last_tick, started = "STOP", 0.0, time.time(), time.time()
    stats = {"received": 0, "rejected": 0, "watchdog_stops": 0}
    driver.set_motion("STOP")
    try:
        while not a.max_runtime or time.time() - started < a.max_runtime:
            try:
                data, (ip, _) = sock.recvfrom(4096)
                msg = json.loads(data)
                cmd = str(msg.get("command", "")).upper()
                ok = (cmd in VALID and (not a.allow or ip == a.allow)
                      and (cmd == "STOP" or msg.get("confidence", 1.0) >= a.min_confidence))
                if ok:
                    stats["received"] += 1
                    last_msg = time.time()
                    if cmd != current:
                        current = cmd
                        driver.set_motion(cmd)
                else:
                    stats["rejected"] += 1
            except socket.timeout:
                pass
            except (json.JSONDecodeError, UnicodeDecodeError):
                stats["rejected"] += 1

            now = time.time()
            if current != "STOP" and now - last_msg > a.timeout:
                print("watchdog: no commands, stopping", flush=True)
                stats["watchdog_stops"] += 1
                current = "STOP"
                driver.set_motion("STOP")
            driver.update(now - last_tick)
            last_tick = now
    except KeyboardInterrupt:
        pass
    finally:
        driver.close()
        print("stopped.", stats, flush=True)


if __name__ == "__main__":
    main()
