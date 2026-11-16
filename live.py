"""
Live control: read the headset's LSL stream, decode, emit motion commands.

  python live.py --model models/me.joblib                 # print commands
  python live.py --model models/me.joblib --udp 5005      # also send JSON over UDP
  python live.py --model models/A01.joblib --score        # with simulate_headset.py: compare to true cues
"""
import argparse
import time

import mne
import numpy as np
from pylsl import StreamInlet, resolve_byprop

from eegmotion import model
from eegmotion.lsl import channel_index, connect
from eegmotion.online import Decoder
from eegmotion.motion import MotionSink
from eegmotion.outputs import PrintSink, UDPSink

mne.set_log_level("ERROR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--robot", help="send commands to the robot: PI_IP or PI_IP:PORT (see robot/pi_receiver.py)")
    ap.add_argument("--udp", type=int, help="also send each decision as JSON to 127.0.0.1:PORT")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run forever)")
    ap.add_argument("--score", action="store_true", help="read true cues from a Markers stream and report accuracy")
    a = ap.parse_args()

    bundle = model.load(a.model)
    cfg = bundle["config"]
    inlet, names, fs = connect(cfg)
    if abs(fs - bundle["fs"]) > 1:
        raise RuntimeError(f"stream is {fs} Hz but model was trained at {bundle['fs']} Hz")
    idx = channel_index(names, bundle["ch_names"])
    dec = Decoder(bundle)
    # MotionSink first so the pose is included in what gets printed / sent over UDP
    sinks = [MotionSink(cfg["online"]["step"]), PrintSink()] + ([UDPSink(port=a.udp)] if a.udp else [])
    if a.robot:
        host, _, port = a.robot.partition(":")
        sinks.append(UDPSink(host, int(port or 5005)))
        print(f"sending commands to robot at {host}:{port or 5005}")

    cue_inlet, cues, all_ts, hits = None, [], [], []
    if a.score:
        m = resolve_byprop("type", "Markers", timeout=5)
        cue_inlet = StreamInlet(m[0]) if m else None
    from eegmotion.model import onset_offset
    win = cfg["online"]["window"]

    t_end = time.time() + a.seconds if a.seconds else float("inf")
    while time.time() < t_end:
        d, ts = inlet.pull_chunk(timeout=0.5)
        if cue_inlet:
            mk, mts = cue_inlet.pull_chunk(timeout=0.0)
            cues += [(t, m[0]) for m, t in zip(mk, mts) if m[0] in cfg["classes"]]
        if not ts:
            continue
        all_ts.extend(ts)
        for decision in dec.push(np.asarray(d)[:, idx]):
            for s in sinks:
                s.send(decision)
            # score the raw (unsmoothed-threshold) label for windows matching the training epoch
            # score windows that line up with the training epoch (tmin..tmax after a cue)
            if cues:
                cue_t, cue = cues[-1]
                cue_s = int(np.searchsorted(all_ts, cue_t))
                lag = (decision["sample"] - cue_s) / fs
                end = onset_offset(cfg, cue) + win  # where the training window for this event ends
                if end <= lag < end + cfg["online"]["step"]:
                    hits.append(decision["label"] == cue)
    if hits:
        print(f"\nlive accuracy on {len(hits)} windows aligned with cues: {np.mean(hits):.1%}")


if __name__ == "__main__":
    main()
