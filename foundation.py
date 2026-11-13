"""
Foundation model from public EEG, then fine-tune on YOUR recordings.

  # now: build from all 9 public subjects
  python foundation.py build data/A0*T.npz --out models/foundation.joblib

  # day 1 with headset: use it directly (live.py re-centers on your data as it streams)
  python live.py --model models/foundation.joblib

  # after recording calibration sessions with record.py:
  python foundation.py finetune models/foundation.joblib recordings/me_*.npz --out models/me.joblib

IMPORTANT for your headset: the foundation and your recordings must use the same
channels. Set headset.channels in config.yaml to the channels your headset has
that also exist in the public data (e.g. [C3, Cz, C4, FC3, FC4, CP3, CP4, Pz]),
then rebuild the foundation.
"""
import argparse
import json
from pathlib import Path

import mne
import numpy as np
from sklearn.model_selection import StratifiedKFold

from eegmotion import data, model, transfer
from eegmotion.config import load_config

mne.set_log_level("ERROR")


def build(a, cfg):
    Cs, ys = [], []
    for f in sorted(a.files):
        X, y = model.epochs(data.load(f, a.loader), cfg)
        C = transfer.covariances(X)
        Cs.append(transfer.recenter(C, transfer.reference(C)))
        ys.append(y)
        print(f"{Path(f).stem}: {len(y)} trials")
    Cs, ys = np.concatenate(Cs), np.concatenate(ys)
    clf = transfer.Classifier().fit(Cs, ys)
    rec = data.load(a.files[0], a.loader).pick(cfg["headset"]["channels"])
    bundle = {"pipeline": transfer.RecenteredPipeline(clf), "classes": list(clf.classes_),
              "fs": rec.fs, "ch_names": rec.ch_names, "config": cfg,
              "source_C": Cs.astype(np.float32), "source_y": ys, "kind": "foundation"}
    model.save(bundle, a.out)
    print(f"foundation: {len(ys)} trials from {len(a.files)} people -> {a.out}")


def finetune(a, cfg):
    base = model.load(a.base)
    if base["fs"] != cfg["headset"]["fs"]:
        print(f"note: foundation at {base['fs']} Hz, config says {cfg['headset']['fs']} Hz; "
              "recordings are used at their own rate, so rebuild the foundation at your headset's rate "
              "(resample the public data) if they differ")
    Cs, ys = base["source_C"].astype(np.float64), base["source_y"]
    own_C, own_y, refs = [], [], []
    for f in a.files:  # each session re-centered on its own mean
        X, y = model.epochs(data.load(f, a.loader), cfg)
        C = transfer.covariances(X)
        M = transfer.reference(C)
        refs.append(M)
        own_C.append(transfer.recenter(C, M)); own_y.append(y)
        print(f"{f}: {len(y)} trials, distance to training before/after alignment "
              f"{transfer.distance_to_training(C, transfer.reference(Cs)):.2f} / "
              f"{transfer.distance_to_training(own_C[-1]):.2f}")
    Co, yo = np.concatenate(own_C), np.concatenate(own_y)

    # cross-validate on own data: foundation-only vs pooled vs own-only
    from pyriemann.tangentspace import TangentSpace
    scores = {"foundation_only": [], "pooled": [], "own_only": []}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Co, yo):
        f0 = transfer.Classifier().fit(Cs, ys)
        pooled = transfer.Classifier().fit(np.concatenate([Cs, Co[tr]]), np.concatenate([ys, yo[tr]]),
                                           sample_weight=np.r_[np.ones(len(ys)), np.full(len(tr), a.own_weight)])
        own = transfer.Classifier().fit(Co[tr], yo[tr])
        for k, m in [("foundation_only", f0), ("pooled", pooled), ("own_only", own)]:
            scores[k].append(float((m.predict(Co[te]) == yo[te]).mean()))
    scores = {k: round(float(np.mean(v)), 3) for k, v in scores.items()}
    best = max(scores, key=scores.get)
    print("5-fold accuracy on your data:", scores, "-> using", best)

    if best == "own_only":
        clf = transfer.Classifier().fit(Co, yo)
    elif best == "pooled":
        clf = transfer.Classifier().fit(np.concatenate([Cs, Co]), np.concatenate([ys, yo]),
                                        sample_weight=np.r_[np.ones(len(ys)), np.full(len(yo), a.own_weight)])
    else:
        clf = transfer.Classifier().fit(Cs, ys)
    M0 = refs[-1]  # start live alignment from the most recent session
    bundle = dict(base, pipeline=transfer.RecenteredPipeline(clf, M0=M0), classes=list(clf.classes_),
                  config=cfg, kind=f"finetuned:{best}", cv=scores)
    model.save(bundle, a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(scores, indent=2))
    print("saved", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("files", nargs="+")
    b.add_argument("--loader", default="bci4_2a"); b.add_argument("--out", default="models/foundation.joblib")
    f = sub.add_parser("finetune"); f.add_argument("base"); f.add_argument("files", nargs="+")
    f.add_argument("--loader", default="npz"); f.add_argument("--own-weight", type=float, default=10.0)
    f.add_argument("--out", default="models/finetuned.joblib")
    for p in (b, f):
        p.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    (build if a.cmd == "build" else finetune)(a, cfg)


if __name__ == "__main__":
    main()
