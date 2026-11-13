"""
Train (and cross-validate) a decoder from one or more recordings.

  # public dataset (now)
  python train.py --loader bci4_2a data/A01T.npz --out models/A01.joblib

  # user calibration sessions (from record.py)
  python train.py recordings/me_*.npz --out models/me.joblib
"""
import argparse
import json

import mne
import numpy as np

from eegmotion import data, model
from eegmotion.config import load_config

mne.set_log_level("ERROR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loader", default="npz", choices=["npz", "bci4_2a", "csv"])
    ap.add_argument("--events", help="events CSV (csv loader)")
    ap.add_argument("--time-col", help="timestamp column in the signal CSV (csv loader)")
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="use only the first part of each recording (keep the rest for a live test)")
    ap.add_argument("--out", default="models/model.joblib")
    a = ap.parse_args()
    cfg = load_config(a.config)

    Xs, ys, rec = [], [], None
    for f in a.files:
        rec = data.load(f, a.loader, events=a.events, fs=cfg["headset"]["fs"], time_col=a.time_col)
        rec = rec.truncate(a.train_fraction)
        print(f, rec)
        X, y = model.epochs(rec, cfg)
        Xs.append(X); ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)

    for kind in ["csp_lda", "riemann"]:
        r = model.evaluate(X, y, kind)
        print(f"{kind:8s} 5-fold accuracy {r['accuracy']:.1%} (chance {r['chance']:.0%}, n={len(y)})")
    r = model.evaluate(X, y, cfg["model"]["type"])
    print("confusion (rows = true):", r["labels"])
    for lab, row in zip(r["labels"], r["confusion"]):
        print(f"  {lab:12s} {row}")

    bundle = model.train(X, y, cfg, rec)
    bundle["cv"] = r
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(bundle, a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps({k: v for k, v in r.items()}, indent=2))
    print("saved", a.out)


if __name__ == "__main__":
    main()
