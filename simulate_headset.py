"""
Headset simulator. Replays a recording as a real-time LSL EEG stream, plus a
marker stream carrying the true cues, for hardware-free end-to-end testing of
record.py and live.py.

  python simulate_headset.py --loader bci4_2a data/A01T.npz --start-fraction 0.75
"""
import argparse
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

from eegmotion import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--loader", default="npz", choices=["npz", "bci4_2a"])
    ap.add_argument("--start-fraction", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=1.0, help=">1 replays faster than real time")
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    rec = data.load(a.file, a.loader)

    info = StreamInfo("SimulatedEEG", "EEG", len(rec.ch_names), rec.fs, "float32", "sim-eeg")
    chans = info.desc().append_child("channels")
    for n in rec.ch_names:
        chans.append_child("channel").append_child_value("label", n)
    eeg = StreamOutlet(info, chunk_size=int(rec.fs / 10))
    markers = StreamOutlet(StreamInfo("SimulatedCues", "Markers", 1, 0, "string", "sim-cues"))
    cues = dict(zip(rec.event_samples.tolist(), rec.event_labels.tolist()))

    start = int(len(rec.signal) * a.start_fraction)
    chunk = int(rec.fs / 10)
    print(f"streaming {rec} from {start / rec.fs:.0f}s at {a.speed}x  (Ctrl+C to stop)", flush=True)
    while True:
        t0, i = local_clock(), start
        while i < len(rec.signal):
            block = rec.signal[i:i + chunk]
            for s in range(i, i + len(block)):
                if s in cues:
                    markers.push_sample([cues[s]])
            eeg.push_chunk(block.tolist())
            i += len(block)
            lag = t0 + (i - start) / rec.fs / a.speed - local_clock()
            if lag > 0:
                time.sleep(lag)
        if not a.loop:
            break
    print("done")


if __name__ == "__main__":
    main()
