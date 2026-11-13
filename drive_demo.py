"""
Offline end-to-end demo: EEG recording -> live decoder -> motion commands ->
simulated robot. Runs as fast as possible (no LSL), writes a trace and an HTML
replay viewable in a browser.

  python drive_demo.py --model models/A01_finetuned.joblib --loader bci4_2a data/A01T.npz \
      --start-fraction 0.75 --seconds 180 --out results/drive
"""
import argparse
import json
from pathlib import Path

import mne
import numpy as np

from eegmotion import data, model
from eegmotion.motion import Agent
from eegmotion.online import Decoder

mne.set_log_level("ERROR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--model", required=True)
    ap.add_argument("--loader", default="npz")
    ap.add_argument("--start-fraction", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--out", default="results/drive")
    a = ap.parse_args()

    bundle = model.load(a.model)
    cfg = bundle["config"]
    rec = data.load(a.file, a.loader).pick(bundle["ch_names"])
    start = int(len(rec.signal) * a.start_fraction)
    stop = min(len(rec.signal), start + int(a.seconds * rec.fs))
    dec, agent = Decoder(bundle), Agent()
    dt = cfg["online"]["step"]
    cue_s = rec.event_samples - start
    cues = [(int(s), str(l)) for s, l in zip(cue_s, rec.event_labels) if 0 <= s < stop - start]
    tmax = cfg["epoch"]["tmax"]

    trace, chunk = [], int(rec.fs / 10)
    for i in range(start, stop, chunk):
        for d in dec.push(rec.signal[i:min(i + chunk, stop)]):
            t = d["sample"] / rec.fs
            prev = [c for c in cues if c[0] <= d["sample"]]
            active = prev[-1] if prev and (d["sample"] - prev[-1][0]) / rec.fs <= 4.0 else None
            trace.append({"t": round(t, 2), "command": d["command"], "label": d["label"],
                          "confidence": d["confidence"], "probs": d["probs"],
                          "cue": cfg["classes"].get(active[1]) if active else None,
                          "pose": agent.apply(d["command"], dt)})

    # summary: decision at the end of each training-aligned window after a cue
    hits = []
    for s, lab in cues:
        at = [x for x in trace if abs(x["t"] - (s / rec.fs + tmax)) < dt / 2 + 1e-9]
        if at:
            hits.append(cfg["classes"][at[0]["label"]] == cfg["classes"][lab])
    summary = {"seconds": round((stop - start) / rec.fs, 1), "decisions": len(trace),
               "cues": len(cues), "accuracy_at_cue_window": round(float(np.mean(hits)), 3) if hits else None,
               "commands_issued": {c: sum(x["command"] == c for x in trace)
                                   for c in list(cfg["classes"].values()) + [cfg["online"]["idle_command"]]}}
    print(summary)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "step": dt, "classes": cfg["classes"], "trace": trace,
               "cues": [{"t": round(s / rec.fs, 2), "command": cfg["classes"][l]} for s, l in cues]}
    (out / "trace.json").write_text(json.dumps(payload))
    tpl = (Path(__file__).parent / "viewer_template.html").read_text()
    (out / "replay.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(payload)))
    print("wrote", out / "trace.json", "and", out / "replay.html")


if __name__ == "__main__":
    main()
