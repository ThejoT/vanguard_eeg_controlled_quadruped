"""
Robot drivers: the hardware abstraction layer for the quadruped.

pi_receiver.py calls, on every loop tick (default 50 Hz):
    driver.set_motion(command)   when the command changes: FORWARD/BACKWARD/LEFT/RIGHT/STOP
    driver.update(dt)            every tick (for gaits that need continuous animation)
and driver.stop() / driver.close() on shutdown or when the safety watchdog fires.

Pick a driver with --driver:
    dryrun                 prints what it would do (no hardware; test the link first)
    gait                   simple trot gait for 8 or 12 hobby servos on a PCA9685 board
                           (template: calibrate robot/robot_config.json per robot)
    mymodule:MyDriver      a custom class, e.g. wrapping an existing locomotion controller
"""
import json
import math
import time
from pathlib import Path


class Driver:
    def start(self):
        pass

    def set_motion(self, command):
        raise NotImplementedError

    def update(self, dt):
        pass

    def stop(self):
        self.set_motion("STOP")

    def close(self):
        self.stop()


class DryRunDriver(Driver):
    def set_motion(self, command):
        print(f"[{time.strftime('%H:%M:%S')}] robot -> {command}", flush=True)


# ---------------------------------------------------------------------------
# Trot gait template for servo quadrupeds (e.g. PCA9685 16-channel servo HAT).
# Each leg has a hip (swing forward/back) and knee (lift), optionally a
# shoulder/abduction joint (12-servo builds). Diagonal pairs move together.
# ---------------------------------------------------------------------------
LEGS = ["front_left", "front_right", "back_left", "back_right"]
DIAGONAL_PHASE = {"front_left": 0.0, "back_right": 0.0, "front_right": 0.5, "back_left": 0.5}

# per command: (stride sign left side, stride sign right side). Turning = legs on
# each side stepping in opposite directions.
STRIDE = {"FORWARD": (1, 1), "BACKWARD": (-1, -1), "LEFT": (-1, 1), "RIGHT": (1, -1), "STOP": (0, 0)}


class ServoBackend:
    """Writes angles to servos. Uses adafruit-circuitpython-servokit if present, else prints."""

    def __init__(self, channels=16, dry=False):
        self.kit = None
        if not dry:
            try:
                from adafruit_servokit import ServoKit
                self.kit = ServoKit(channels=channels)
            except Exception as e:  # not on a Pi / library missing
                print(f"servo board not available ({e}); printing angles instead")
        self.last_print = 0

    def write(self, angles):
        if self.kit:
            for ch, a in angles.items():
                self.kit.servo[ch].angle = max(0, min(180, a))
        elif time.time() - self.last_print > 0.5:
            self.last_print = time.time()
            print("servos:", {k: round(v) for k, v in sorted(angles.items())}, flush=True)


class GaitDriver(Driver):
    def __init__(self, config_path=None, dry=False):
        path = Path(config_path or Path(__file__).with_name("robot_config.json"))
        self.cfg = json.loads(path.read_text())
        self.servos = ServoBackend(dry=dry)
        self.phase, self.command, self.blend = 0.0, "STOP", 0.0
        self.stride = (0, 0)  # last moving stride, kept while easing down to a stop

    def start(self):
        self.servos.write(self._pose(0.0, 0.0))  # stand

    def set_motion(self, command):
        self.command = command
        if command != "STOP":
            self.stride = STRIDE[command]
        print(f"gait -> {command}", flush=True)

    def update(self, dt):
        g = self.cfg["gait"]
        target = 0.0 if self.command == "STOP" else 1.0
        # ease in/out so the robot doesn't jerk between commands
        self.blend += (target - self.blend) * min(1.0, dt / g["ramp_s"])
        if self.blend > 0.01:
            self.phase = (self.phase + dt * g["frequency_hz"]) % 1.0
        self.servos.write(self._pose(self.phase, self.blend))

    def _pose(self, phase, amount):
        g, out = self.cfg["gait"], {}
        sl, sr = self.stride if amount > 0 else (0, 0)
        for leg in LEGS:
            j = self.cfg["legs"][leg]
            p = (phase + DIAGONAL_PHASE[leg]) % 1.0
            side = sl if "left" in leg else sr
            swing = math.sin(2 * math.pi * p)                    # hip: forward/back
            lift = max(0.0, math.sin(2 * math.pi * p))           # knee: lift during swing half
            hip = j["hip_neutral"] + j["hip_dir"] * side * g["stride_deg"] * swing * amount
            knee = j["knee_neutral"] + j["knee_dir"] * g["lift_deg"] * lift * amount * (1 if side else 0)
            out[j["hip_channel"]] = hip
            out[j["knee_channel"]] = knee
            if "shoulder_channel" in j:
                out[j["shoulder_channel"]] = j["shoulder_neutral"]
        return out

    def stop(self):
        self.command = "STOP"
        for _ in range(int(self.cfg["gait"]["ramp_s"] * 50)):  # settle to standing
            self.update(0.02)
            time.sleep(0.02)


def load_driver(name, config=None, dry=False):
    if name == "dryrun":
        return DryRunDriver()
    if name == "gait":
        return GaitDriver(config, dry=dry)
    if ":" in name:  # custom "module:Class"
        import importlib
        mod, cls = name.split(":")
        return getattr(importlib.import_module(mod), cls)()
    raise ValueError(f"unknown driver {name}")
