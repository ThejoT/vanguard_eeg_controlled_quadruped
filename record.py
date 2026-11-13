"""
Calibration recorder. Presents randomized motor-imagery cues in the terminal,
records the EEG LSL stream with cue markers, and saves a labelled Recording (.npz).

  python record.py --subject me --session 1

Each trial: REST (relax, look at a fixed point) -> cue (imagine the movement,
kinaesthetically - feel it, don't just picture it - without moving).
Subject guidance: remain still, keep the jaw relaxed, avoid blinking during cues.
Record 2-4 sessions on different days before trusting the model.
"""
import argparse
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from eegmotion.config import load_config
from eegmotion.data import Recording
from eegmotion.lsl import connect

PROMPTS = {"left_hand": "<<<  imagine squeezing your LEFT hand",
           "right_hand": ">>>  imagine squeezing your RIGHT hand",
           "feet": " ^   imagine pressing down with BOTH FEET",
           "tongue": " v   imagine pressing your TONGUE to the roof of your mouth",
           "rest": "  +  relax, keep looking at the cross"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--subject", default="me")
    ap.add_argument("--session", default="1")
    ap.add_argument("--trials", type=int, help="override trials per class")
    ap.add_argument("--rest", type=float); ap.add_argument("--cue", type=float)
    ap.add_argument("--out", default="recordings")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cal = cfg["calibration"]
    n, rest, cue = a.trials or cal["trials_per_class"], a.rest or cal["rest"], a.cue or cal["cue"]

    inlet, names, fs = connect(cfg)
    chunks, stamps, stop = [], [], threading.Event()

    def reader():
        while not stop.is_set():
            d, ts = inlet.pull_chunk(timeout=0.2)
            if ts:
                chunks.append(np.asarray(d, np.float32)); stamps.extend(ts)

    th = threading.Thread(target=reader, daemon=True); th.start()
    plan = [c for c in cfg["classes"] for _ in range(n)]
    random.shuffle(plan)
    events = []
    print(f"\n{len(plan)} trials, ~{len(plan) * (rest + cue) / 60:.1f} min. Starting in 3 s...\n")
    time.sleep(3)
    try:
        for k, lab in enumerate(plan, 1):
            print(f"[{k:3d}/{len(plan)}] {PROMPTS['rest']}", flush=True)
            events.append((inlet_time(), "rest"))  # enables an optional rest -> STOP class
            time.sleep(rest + random.uniform(0, 0.5))
            print(f"[{k:3d}/{len(plan)}] {PROMPTS.get(lab, lab)}", flush=True)
            events.append((inlet_time(), lab))
            time.sleep(cue)
    except KeyboardInterrupt:
        print("\nstopped early; saving what was recorded")
    time.sleep(0.5); stop.set(); th.join()

    sig = np.concatenate(chunks)
    ts = np.asarray(stamps)
    samples = np.searchsorted(ts, [t for t, _ in events])
    rec = Recording(sig, fs, names, samples, np.array([l for _, l in events], dtype="<U32"))
    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"{a.subject}_s{a.session}_{datetime.now():%Y%m%d_%H%M}.npz"
    rec.save(path)
    print(f"saved {path}: {rec}")


def inlet_time():
    from pylsl import local_clock
    return local_clock()


if __name__ == "__main__":
    main()
