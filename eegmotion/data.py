"""
One standard in-memory format for every data source.

    Recording.signal         (n_samples, n_channels) float32, microvolts
    Recording.fs             sampling rate (Hz)
    Recording.ch_names       list of channel names
    Recording.event_samples  sample index of each cue onset
    Recording.event_labels   class name of each cue (e.g. "left_hand")

Write a loader that returns a Recording and the rest of the pipeline works.
Loaders included: BCI IV 2a (.npz port), our own .npz, and CSV (what most
headset apps export: OpenBCI GUI, Muse apps, BrainFlow, etc.).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Recording:
    signal: np.ndarray
    fs: float
    ch_names: list
    event_samples: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    event_labels: np.ndarray = field(default_factory=lambda: np.zeros(0, "<U32"))

    def pick(self, names):
        """Return a copy restricted to `names` (in that order). None = all."""
        if names is None:
            return self
        missing = [n for n in names if n not in self.ch_names]
        if missing:
            raise ValueError(f"channels not in recording: {missing}. Available: {self.ch_names}")
        idx = [self.ch_names.index(n) for n in names]
        return Recording(self.signal[:, idx], self.fs, list(names),
                         self.event_samples, self.event_labels)

    def truncate(self, fraction):
        """Keep the first `fraction` of the recording (and the events inside it)."""
        n = int(len(self.signal) * fraction)
        keep = self.event_samples < n
        return Recording(self.signal[:n], self.fs, self.ch_names,
                         self.event_samples[keep], self.event_labels[keep])

    def save(self, path):
        np.savez_compressed(path, signal=self.signal.astype(np.float32), fs=self.fs,
                            ch_names=np.array(self.ch_names, dtype="<U32"),
                            event_samples=np.asarray(self.event_samples, dtype=np.int64),
                            event_labels=np.asarray(self.event_labels, dtype="<U32"))

    def __repr__(self):
        labels, counts = np.unique(self.event_labels, return_counts=True)
        return (f"Recording({self.signal.shape[0] / self.fs:.0f}s, {len(self.ch_names)} ch @ "
                f"{self.fs:g} Hz, events={dict(zip(labels.tolist(), counts.tolist()))})")


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return Recording(d["signal"], float(d["fs"]), d["ch_names"].tolist(),
                     d["event_samples"], d["event_labels"])


BCI4_2A_CHANNELS = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
                    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
BCI4_2A_CUES = {769: "left_hand", 770: "right_hand", 771: "feet", 772: "tongue"}


def load_bci4_2a(path, drop_artifacts=True, rest=True):
    """BCI Competition IV 2a, numpy port (github.com/bregydoc/bcidatasetIV2a). Use the *T files.

    Each trial starts with a 2 s fixation period before the cue (Tangermann et al., 2012).
    With rest=True, a "rest" event is added at each trial start so that period can train
    an idle class."""
    d = np.load(path, allow_pickle=False)
    sig = np.nan_to_num(d["s"][:, :22]).astype(np.float32)
    etyp, epos, art = d["etyp"].ravel(), d["epos"].ravel(), d["artifacts"].ravel()
    samples, labels, trial = [], [], -1
    for code, pos in zip(etyp, epos):
        if code == 768:
            trial += 1
            if rest and not (drop_artifacts and art[trial]):
                samples.append(pos)
                labels.append("rest")
        if code in BCI4_2A_CUES and not (drop_artifacts and art[trial]):
            samples.append(pos)
            labels.append(BCI4_2A_CUES[code])
    return Recording(sig, 250.0, list(BCI4_2A_CHANNELS),
                     np.array(samples), np.array(labels, dtype="<U32"))


def load_csv(signal_csv, events_csv, fs, time_col=None):
    """
    signal_csv: header row of channel names, one row per sample. An optional
                timestamp column (named by `time_col`) is used to place events.
    events_csv: columns `label` and either `sample` (row index) or `time`
                (same clock as time_col).
    """
    import pandas as pd
    sig = pd.read_csv(signal_csv)
    ev = pd.read_csv(events_csv)
    times = None
    if time_col:
        times = sig.pop(time_col).to_numpy()
    if "sample" in ev:
        samples = ev["sample"].to_numpy(int)
    elif times is not None:
        samples = np.searchsorted(times, ev["time"].to_numpy())
    else:
        raise ValueError("events need a `sample` column, or pass time_col and give events a `time` column")
    return Recording(sig.to_numpy(np.float32), float(fs), list(sig.columns),
                     samples, ev["label"].astype(str).to_numpy("<U32"))


def load(path, loader="npz", **kw):
    path = Path(path)
    if loader == "bci4_2a":
        return load_bci4_2a(path)
    if loader == "npz":
        return load_npz(path)
    if loader == "csv":
        return load_csv(path, kw["events"], kw["fs"], kw.get("time_col"))
    raise ValueError(f"unknown loader {loader}")
