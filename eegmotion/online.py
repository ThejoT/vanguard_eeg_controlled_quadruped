"""
Live decoder: push raw EEG chunks in, get motion commands out.
Has no knowledge of LSL or any headset; feed it from anywhere.
"""
import numpy as np

from .preprocess import StreamFilter


class Decoder:
    def __init__(self, bundle, cfg=None):
        self.cfg = cfg or bundle["config"]
        self.pipe = bundle["pipeline"]
        self.classes = bundle["classes"]
        self.fs = bundle["fs"]
        on = self.cfg["online"]
        self.win = int(round(on["window"] * self.fs))
        self.step = int(round(on["step"] * self.fs))
        self.alpha, self.thr, self.hold = on["smoothing"], on["threshold"], on["hold"]
        self.idle = on["idle_command"]
        self.cmd_map = self.cfg["classes"]
        n_ch = len(bundle["ch_names"])
        self.filter = StreamFilter(self.fs, self.cfg["preprocessing"]["band"],
                                   self.cfg["preprocessing"]["notch"], n_ch)
        self.buf = np.zeros((0, n_ch))
        self.since = 0
        self.n_seen = 0  # samples consumed so far (decision['sample'] = window end)
        self.smoothed = None
        self.streak_label, self.streak = None, 0

    def push(self, chunk):
        """chunk: (n_samples, n_channels) raw EEG in the model's channel order.
        Returns a list of decisions (one per `step` of new data once the buffer is full)."""
        if len(chunk) == 0:
            return []
        f, out, i = self.filter(chunk), [], 0
        while i < len(f):
            take = min(self.step - self.since, len(f) - i)
            self.buf = np.vstack([self.buf, f[i:i + take]])[-self.win:]
            self.since += take
            self.n_seen += take
            i += take
            if self.since >= self.step:
                self.since = 0
                if len(self.buf) == self.win:
                    out.append(self._decide())
        return out

    def _decide(self):
        p = self.pipe.predict_proba(self.buf.T[None])[0]
        self.smoothed = p if self.smoothed is None else self.alpha * self.smoothed + (1 - self.alpha) * p
        k = int(self.smoothed.argmax())
        label, conf = self.classes[k], float(self.smoothed[k])
        if conf >= self.thr and label == self.streak_label:
            self.streak += 1
        elif conf >= self.thr:
            self.streak_label, self.streak = label, 1
        else:
            self.streak_label, self.streak = None, 0
        command = self.cmd_map[label] if self.streak >= self.hold else self.idle
        return {"command": command, "label": label, "confidence": round(conf, 3),
                "sample": self.n_seen, "probs": {c: round(float(v), 3) for c, v in zip(self.classes, self.smoothed)}}
