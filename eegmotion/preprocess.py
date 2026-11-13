"""
Causal filtering, used identically offline (training) and online (live).
Using the same causal filter in both places means the live decoder sees data
that looks exactly like what the model was trained on.
"""
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos


def design(fs, band, notch=None):
    sos = butter(4, band, btype="band", fs=fs, output="sos")
    if notch:
        b, a = iirnotch(notch, Q=30, fs=fs)
        sos = np.vstack([tf2sos(b, a), sos])
    return sos


class StreamFilter:
    """Filter that keeps its state between chunks (for live data)."""

    def __init__(self, fs, band, notch, n_channels):
        self.sos = design(fs, band, notch)
        self.zi = np.repeat(sosfilt_zi(self.sos)[:, :, None], n_channels, axis=2) * 0.0
        self.primed = False

    def __call__(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float64)
        if not self.primed:  # start from the first sample's level to avoid a big transient
            self.zi = sosfilt_zi(self.sos)[:, :, None] * chunk[0][None, None, :]
            self.primed = True
        out, self.zi = sosfilt(self.sos, chunk, axis=0, zi=self.zi)
        return out


def filter_recording(rec, cfg):
    f = StreamFilter(rec.fs, cfg["preprocessing"]["band"], cfg["preprocessing"]["notch"],
                     rec.signal.shape[1])
    return f(rec.signal)
