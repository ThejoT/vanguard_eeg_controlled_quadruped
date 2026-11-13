# EEG → motion

Turns motor-imagery EEG (imagining moving your left hand, right hand, feet or
tongue) into motion commands: LEFT, RIGHT, FORWARD, BACKWARD, or STOP.
The full software stack works today on public EEG data. When you get a headset,
its live data plugs into the same pipeline: it gets aligned to the training data,
and the model is fine-tuned on your own recordings.

```
 public EEG ──► foundation model ─┐
                                  ├─► live decoder ─► command ─► motion (sim / robot / UDP)
 your headset ─► LSL stream ──────┘        ▲
       └─► record.py ─► calibration ─► fine-tune
```

## Setup
```
pip install -r requirements.txt
git clone https://github.com/bregydoc/bcidatasetIV2a data_src   # BCI Competition IV 2a, ~600 MB
mkdir data && cp data_src/A0*T.npz data/
```

## Now: develop on public EEG
| Step | Command |
|---|---|
| Build a foundation model from 9 people | `python foundation.py build data/A0*T.npz --out models/foundation.joblib` |
| How well it transfers to a new person | `python transfer_eval.py data/A0*T.npz` |
| Pretend to be a headset (LSL stream) | `python simulate_headset.py --loader bci4_2a data/A01T.npz` |
| Live control from that stream | `python live.py --model models/foundation.joblib --score` |
| Send commands to your robot/game code | `python live.py --model ... --udp 5005` (JSON with command, confidence, pose) |
| Offline replay + visual | `python drive_demo.py data/A01T.npz --loader bci4_2a --model ... --out results/drive` → open `results/drive/replay.html` |
| Single-person model (baseline) | `python train.py --loader bci4_2a data/A01T.npz --out models/A01.joblib` |

## Later: your headset
1. **Stream it over LSL.** OpenBCI GUI (Networking → LSL), BrainFlow, `muselsl`, Emotiv/Unicorn/g.tec LSL apps all work.
2. **Edit `config.yaml`:** `headset.fs`, and `headset.channels` = the channels your headset has
   that also exist in the public data (C3, Cz, C4 are the essential ones). Set `notch` (60 Hz in the US).
3. **Rebuild the foundation** with those channels: `python foundation.py build data/A0*T.npz`.
   If your headset's sampling rate isn't 250 Hz, resample the public data to match first.
4. **Day 1:** `python live.py --model models/foundation.joblib`. It re-centers your
   live EEG onto the training distribution as it streams, so it needs no calibration to start.
5. **Calibrate:** `python record.py --subject me --session 1` (~10 min). Do 2–4 sessions on different days.
6. **Fine-tune:** `python foundation.py finetune models/foundation.joblib recordings/me_*.npz --out models/me.joblib`.
   It reports how far your data sits from the training data before and after alignment, and compares
   foundation-only, pooled, and own-data-only models on your data, then keeps the best.
7. **Add a rest class** (uncomment `rest: STOP` in config) so it holds still when you aren't imagining anything.
8. Repeat 5–6 as you collect more data. It keeps improving.

Your own CSV exports work too: `train.py --loader csv signal.csv --events events.csv --time-col timestamp`.

## Results so far (BCI IV 2a, 4 classes, chance 25%)
| Scenario | Accuracy |
|---|---|
| Model trained on that person (cross-validated, 9-person mean) | 68% |
| Other people's model, raw (no alignment) | 38% |
| Other people's model + re-centering, zero labelled calibration | 52% (offline) / 68% live on subject A01 |
| + 20 calibration trials per class | 58% |
| Fine-tuned on one calibration session (A01, live) | 76% |

People vary a lot (40–86%). Some produce weak motor-imagery signals at first, and practice
with feedback helps.

## Headset notes
- Motor imagery needs electrodes over the motor cortex: **C3, Cz, C4** at minimum, ideally
  FC3/FC4/CP3/CP4 too. Gel or good dry electrodes over hair.
- **Muse** (TP9, AF7, AF8, TP10) has no motor-cortex channels, so this approach won't work with it.
- OpenBCI Cyton (8 ch, or 16 with Daisy) with an electrode cap is the common hobby/research choice.
- Feet vs. tongue (FORWARD vs. BACKWARD) is the hardest pair to separate. If two commands
  are enough for a first version, LEFT/RIGHT is the most reliable.

## Code map
`eegmotion/data.py` standard Recording format + loaders ·
`preprocess.py` causal filtering (identical offline and live) ·
`model.py` epoching, CSP/Riemannian pipelines ·
`transfer.py` re-centering, online alignment, distance-to-training ·
`online.py` live decoder (smoothing, threshold, hold → command) ·
`motion.py` command → pose (swap in your motor driver) ·
`outputs.py` print / UDP sinks · `lsl.py` headset stream I/O
