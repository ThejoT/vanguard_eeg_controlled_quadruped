"""
How well does a model trained on OTHER people work on a NEW person, and how
fast does it improve with that person's own calibration data?

Leave-one-subject-out on BCI IV 2a: for each subject, train on the other 8,
then test on that subject's second half, with 0, 5, 10, 20 labelled
calibration trials per class taken from their first half.

This is a rehearsal of what will happen when your headset arrives:
you are the "new person".

  python transfer_eval.py data/A0*T.npz --out results/transfer.json
"""
import argparse
import json
from pathlib import Path

import mne
import numpy as np

from eegmotion import data, model, transfer
from eegmotion.config import load_config

mne.set_log_level("ERROR")
K_LIST = [0, 5, 10, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results/transfer.json")
    a = ap.parse_args()
    cfg = load_config(a.config)

    subj = {}
    for f in sorted(a.files):
        X, y = model.epochs(data.load_bci4_2a(f), cfg)
        subj[Path(f).stem] = (transfer.covariances(X), y)
        print("loaded", Path(f).stem, len(y))

    rng = np.random.default_rng(0)
    res = {}
    for tgt, (Ct, yt) in subj.items():
        half = len(yt) // 2
        Cpool, ypool, Ctest, ytest = Ct[:half], yt[:half], Ct[half:], yt[half:]

        # source model: other subjects, each re-centered on its own mean
        Cs = np.concatenate([transfer.recenter(C, transfer.reference(C)) for s, (C, _) in subj.items() if s != tgt])
        ys = np.concatenate([y for s, (_, y) in subj.items() if s != tgt])
        # no-alignment baseline: same data, raw
        Cs_raw = np.concatenate([C for s, (C, _) in subj.items() if s != tgt])

        # target reference from its first half, UNLABELLED (like the first minutes of using the headset)
        Mt = transfer.reference(Cpool)
        Ctest_rc = transfer.recenter(Ctest, Mt)
        Cpool_rc = transfer.recenter(Cpool, Mt)

        r = {"distance_raw": transfer.distance_to_training(Ctest, transfer.reference(Cs_raw)),
             "distance_aligned": transfer.distance_to_training(Ctest_rc)}
        from sklearn.pipeline import make_pipeline
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
        raw = make_pipeline(TangentSpace(), LogisticRegression(max_iter=3000)).fit(Cs_raw, ys)
        r["others_no_alignment"] = float((raw.predict(Ctest) == ytest).mean())

        for k in K_LIST:
            idx = np.concatenate([rng.permutation(np.where(ypool == c)[0])[:k] for c in np.unique(ypool)]) if k else np.array([], int)
            # pooled: other people + k own trials (weighted up so they count)
            C_tr = np.concatenate([Cs, Cpool_rc[idx]]) if k else Cs
            y_tr = np.concatenate([ys, ypool[idx]]) if k else ys
            w = np.concatenate([np.ones(len(ys)), np.full(len(idx), 10.0)]) if k else None
            clf = transfer.Classifier().fit(C_tr, y_tr, sample_weight=w)
            r[f"pooled_k{k}"] = float((clf.predict(Ctest_rc) == ytest).mean())
            if k:  # own data only
                own = make_pipeline(TangentSpace(), LogisticRegression(max_iter=3000)).fit(Cpool[idx], ypool[idx])
                r[f"own_only_k{k}"] = float((own.predict(Ctest) == ytest).mean())
        own_all = make_pipeline(TangentSpace(), LogisticRegression(max_iter=3000)).fit(Cpool, ypool)
        r["own_only_all"] = float((own_all.predict(Ctest) == ytest).mean())
        res[tgt] = r
        print(tgt, {k: round(v, 2) for k, v in r.items()})

    keys = list(next(iter(res.values())).keys())
    res["_mean"] = {k: round(float(np.mean([res[s][k] for s in subj])), 3) for k in keys}
    print("\nMEAN", res["_mean"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
