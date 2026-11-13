"""
Turn discrete commands into motion. A simple differential-drive model
(wheelchair / small robot): FORWARD/BACKWARD move along the heading,
LEFT/RIGHT rotate in place, STOP holds still. Replace `Agent.apply` with a
hardware driver (motor controller, ROS cmd_vel, Arduino serial...).
"""
import math
from dataclasses import dataclass

VELOCITY = {  # (linear m/s, angular deg/s)
    "FORWARD": (0.5, 0.0),
    "BACKWARD": (-0.3, 0.0),
    "LEFT": (0.0, 45.0),
    "RIGHT": (0.0, -45.0),
    "STOP": (0.0, 0.0),
}


@dataclass
class Agent:
    x: float = 0.0
    y: float = 0.0
    heading: float = 90.0  # degrees, 90 = "up"

    def apply(self, command, dt):
        v, w = VELOCITY.get(command, (0.0, 0.0))
        self.heading = (self.heading + w * dt) % 360
        self.x += v * dt * math.cos(math.radians(self.heading))
        self.y += v * dt * math.sin(math.radians(self.heading))
        return self.pose()

    def pose(self):
        return {"x": round(self.x, 3), "y": round(self.y, 3), "heading": round(self.heading, 1)}


class MotionSink:
    """Output sink for live.py: integrates commands into a pose."""

    def __init__(self, dt):
        self.agent, self.dt = Agent(), dt

    def send(self, d):
        d["pose"] = self.agent.apply(d["command"], self.dt)
