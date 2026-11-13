"""Lab Streaming Layer helpers. Most EEG headsets can stream over LSL:
OpenBCI GUI (Networking tab -> LSL), BrainFlow, muselsl, Emotiv (via LSL plugin),
g.tec, Brain Products, Neurosity, Unicorn, etc."""
from pylsl import StreamInlet, resolve_byprop


def connect(cfg, timeout=10.0):
    h = cfg["headset"]
    streams = (resolve_byprop("name", h["stream_name"], timeout=timeout) if h.get("stream_name")
               else resolve_byprop("type", h["stream_type"], timeout=timeout))
    if not streams:
        raise RuntimeError(f"no LSL stream of type {h['stream_type']!r} found. Is the headset streaming?")
    inlet = StreamInlet(streams[0], max_buflen=60)
    info = inlet.info()
    names, ch = [], info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        names.append(ch.child_value("label") or f"ch{len(names)}")
        ch = ch.next_sibling()
    print(f"connected: {info.name()} ({info.channel_count()} ch @ {info.nominal_srate():g} Hz)")
    return inlet, names, info.nominal_srate()


def channel_index(stream_names, wanted):
    """Map configured channel names to stream column indices."""
    if wanted is None:
        return list(range(len(stream_names)))
    missing = [w for w in wanted if w not in stream_names]
    if missing:
        raise RuntimeError(f"stream lacks channels {missing}; it has {stream_names}. "
                           "Edit headset.channels in config.yaml")
    return [stream_names.index(w) for w in wanted]
