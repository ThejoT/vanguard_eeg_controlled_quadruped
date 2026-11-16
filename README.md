# Vanguard: EEG-Controlled Quadruped

A brain-computer interface that drives a quadruped robot with motor imagery. Imagined
left-hand, right-hand, feet and tongue movements are decoded from EEG in real time and
mapped to `LEFT`, `RIGHT`, `FORWARD` and `BACKWARD` locomotion commands.

The decoder is trained on public EEG data and adapts to new users and headsets through
Riemannian re-centering, so it runs with no calibration and improves as user data is
collected.

## Architecture

```
Offline     public EEG ──► foundation model ──► fine-tuned model
                                  ▲                    │
            user sessions ────────┘                    │
                                                       ▼
Online      headset ──► filter ──► covariance ──► re-center ──► classify ──► smooth / threshold
(LSL)                   8–30 Hz    2 s window                                       │
                                                                                    │ UDP
Robot       servos ◄── gait driver ◄── receiver + watchdog ◄────────────────────────┘
(Pi)
```

| Component | Description |
| --- | --- |
| Signal processing | Causal band-pass and notch filters shared by training and inference |
| Features | OAS-shrinkage covariance matrices of 2 s windows |
| Classifier | Riemannian tangent space + logistic regression (CSP + LDA baseline included) |
| Transfer | Per-session re-centering (Zanini et al., 2018) with an online running mean |
| Control | Rest class mapped to `STOP`, probability smoothing, confidence threshold and hold time |
| Robot link | JSON over UDP to a Raspberry Pi receiver with a 1 s safety watchdog |

## Results

Evaluated on [BCI Competition IV 2a](https://www.bbci.de/competition/iv/) (9 subjects,
22 channels, 250 Hz, 4 classes, chance = 25%).

| Setting | Accuracy |
| --- | --- |
| Within-subject, 5-fold CV, CSP + LDA | 64.3% |
| Within-subject, 5-fold CV, Riemannian | 68.1% (range 40–86%) |
| Cross-subject, no alignment | 38.1% |
| Cross-subject, re-centered, zero calibration | 52.3% |
| Cross-subject, re-centered, + 20 trials/class | 57.5% |
| Live stream, unseen subject (A01), zero calibration | 68% |
| Live stream, A01 after fine-tuning on one session | 76% |

**Idle behaviour.** A rest class, trained on the 2 s fixation period before each cue, cuts
movement while the user is idle from 48–70% of the time to 21–36% (subjects A01, A03, A07,
held-out data), while matching or improving correct movement during imagery (56–71%).

Cross-subject figures are leave-one-subject-out means. Live figures come from replaying
held-out data in real time through the full LSL pipeline.

## Repository structure

```
eegmotion/            core library
  data.py             Recording format and loaders (BCI IV 2a, NPZ, CSV)
  preprocess.py       causal filtering
  model.py            epoching, CSP and Riemannian pipelines
  transfer.py         re-centering and online alignment
  online.py           streaming decoder
  motion.py           command-to-pose model
  lsl.py, outputs.py  stream I/O and command sinks
hiro/                 Arduino sketches for the HIRO quadruped (Pico 2 W)
  hiro_wifi_control/  Wi-Fi UDP command receiver with safety watchdog
  hiro_keyboard_test/ single-step movement and gait tuning over USB serial
robot/                Raspberry Pi side (standard library only)
  pi_receiver.py      UDP receiver with watchdog
  drivers.py          dry-run, PCA9685 trot gait, custom driver interface
  keyboard_teleop.py  manual control for testing
train.py              within-subject training and evaluation
foundation.py         cross-subject model build and fine-tuning
transfer_eval.py      leave-one-subject-out evaluation
record.py             calibration session recorder
live.py               real-time decoding and robot control
simulate_headset.py   replays a recording as an LSL stream
drive_demo.py         offline replay with HTML visualization
config.yaml           device, class and decoder settings
```

## Installation

```bash
pip install -r requirements.txt
git clone https://github.com/bregydoc/bcidatasetIV2a data_src
mkdir -p data && cp data_src/A0*T.npz data/
```

The Raspberry Pi needs only `robot/`, Python 3 and, for servo control,
`adafruit-circuitpython-servokit`.

## Usage

**Train and evaluate**

```bash
python foundation.py build data/A0*T.npz --out models/foundation.joblib
python transfer_eval.py data/A0*T.npz
python train.py --loader bci4_2a data/A01T.npz --out models/A01.joblib
```

**Run without hardware**

```bash
python simulate_headset.py --loader bci4_2a data/A01T.npz --start-fraction 0.75
python live.py --model models/foundation.joblib --score
python drive_demo.py data/A01T.npz --loader bci4_2a --model models/A01.joblib --out results/drive
```

**Run with a headset and the HIRO robot**

Flash `hiro/hiro_wifi_control` to HIRO (Arduino IDE, board "Raspberry Pi Pico 2W"). By
default HIRO starts its own access point, `HIRO-Vanguard`, at 192.168.4.1; connect the
workstation to it and run:

```bash
python robot/keyboard_teleop.py                                  # keyboard control
python live.py --model models/foundation.joblib --robot 192.168.4.1   # EEG control
```

To join an existing 2.4 GHz network instead, copy `secrets.h.example` to `secrets.h` with
the network credentials before flashing; HIRO then prints its assigned address on the
serial monitor (115200 baud).

**Run with a headset and a Raspberry Pi robot**

```bash
# Raspberry Pi
python3 robot/pi_receiver.py --driver gait

# workstation
python record.py --subject s01 --session 1
python foundation.py finetune models/foundation.joblib recordings/s01_*.npz --out models/s01.joblib
python live.py --model models/s01.joblib --robot <pi-ip>
```

Both robot targets accept the same JSON-over-UDP messages on port 5005.

Any EEG device that publishes an LSL stream is supported. Set the sampling rate,
channel names and mains frequency in `config.yaml`, and rebuild the foundation model on
the channels shared with the public dataset.

## Hardware

**EEG.** Motor imagery requires coverage of the sensorimotor cortex: C3, Cz and C4 at
minimum, ideally with FC3/FC4/CP3/CP4. The reference platform is an OpenBCI Cyton
(8 channels, 250 Hz) with gel electrodes. Headsets without central electrodes
(Muse, Emotiv EPOC X) are not suitable.

**Robot.** The primary platform is the Takamori HIRO quadruped (Raspberry Pi Pico 2 W,
8 servos), driven by `hiro/hiro_wifi_control` using the [HIRO library](https://github.com/TakamoriRobot/HIRO).
A Raspberry Pi target is also provided. For that target, the included gait driver drives
hobby servos on a PCA9685 board and is configured in `robot/robot_config.json`; other
platforms are supported by implementing `set_motion`, `update` and `stop` in a custom
driver class.

**Safety.** The receiver stops the robot if no command arrives within the watchdog
timeout, can reject low-confidence commands and unknown senders, and returns to a
standing pose on shutdown.

## Roadmap

- [ ] Validate on physical EEG hardware and the quadruped
- [x] Rest class for idle detection (public data; per-user calibration pending)
- [ ] Resampling of public data to non-250 Hz headsets
- [ ] Additional training data (PhysioNet EEGMMIDB, 109 subjects)
- [ ] Automated tests and CI

## References

- Tangermann et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55.
- Zanini et al. (2018). Transfer learning: a Riemannian geometry framework with applications to brain-computer interfaces. *IEEE TBME*, 65(5), 1107–1116.
- Barachant et al. (2012). Multiclass brain-computer interface classification by Riemannian geometry. *IEEE TBME*, 59(4), 920–928.
- Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization and desynchronization. *Clinical Neurophysiology*, 110(11), 1842–1857.
