import joblib
import numpy as np
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from .preprocess import filter_recording


def build(kind):
    if kind == "riemann":
        return make_pipeline(Covariances("oas"), TangentSpace(),
                             LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    if kind == "csp_lda":
        return make_pipeline(CSP(n_components=8, log=True, reg="ledoit_wolf"),
                             LinearDiscriminantAnalysis())
    raise ValueError(kind)


def onset_offset(cfg, label):
    """Seconds from an event to the start of its training window. Imagery windows start
    epoch.tmin after the cue; rest windows start epoch.rest_tmin after the rest onset."""
    return cfg["epoch"].get("rest_tmin", 0.0) if label == "rest" else cfg["epoch"]["tmin"]


def epochs(rec, cfg):
    """Cut labelled windows after each event. Only labels listed in cfg['classes'] are kept."""
    rec = rec.pick(cfg["headset"]["channels"])
    filt = filter_recording(rec, cfg)
    n = int(round(cfg["online"]["window"] * rec.fs))
    X, y = [], []
    for s, lab in zip(rec.event_samples, rec.event_labels):
        a = int(round(onset_offset(cfg, lab) * rec.fs))
        if lab in cfg["classes"] and 0 <= s + a and s + a + n <= len(filt):
            X.append(filt[s + a:s + a + n].T)
            y.append(lab)
    if not X:
        raise ValueError("no usable trials: check that event labels match the keys under `classes`")
    return np.array(X), np.array(y)


def evaluate(X, y, kind, folds=5):
    pred = cross_val_predict(build(kind), X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0))
    labels = sorted(str(v) for v in set(y))
    return {"accuracy": float((pred == y).mean()), "chance": 1 / len(labels), "labels": labels,
            "confusion": confusion_matrix(y, pred, labels=labels).tolist()}


def train(X, y, cfg, rec):
    pipe = build(cfg["model"]["type"]).fit(X, y)
    return {"pipeline": pipe, "classes": list(pipe.classes_), "fs": rec.fs,
            "ch_names": cfg["headset"]["channels"] or rec.ch_names, "config": cfg}


def save(bundle, path):
    joblib.dump(bundle, path)


def load(path):
    return joblib.load(path)
