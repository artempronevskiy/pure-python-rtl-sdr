# -*- coding: utf-8 -*-
"""
Mock RTL-TCP Server with FM Modulator – hardware-free test source
=================================================================

Generates properly FM-modulated I/Q samples (uint8, interleaved) that the
decoder can successfully demodulate into actual audio.

Audio sources (pick one via CLI):
  --wav FILE    Read a mono/stereo 16-bit PCM WAV file and FM-modulate it.
                This gives real human-speech output from the decoder.
  --tone FREQ   Emit a pure sine tone at FREQ Hz (default: 1000 Hz).
                Useful for verifying the DSP gain / frequency response.
  (default)     Synthetic voice: overlapping harmonics with amplitude
                envelope mimicking a human vowel.  Sounds like humming.

Usage
-----
Terminal 1:  python mock_rtl_tcp.py                      # synthetic voice
             python mock_rtl_tcp.py --tone 440            # A4 test tone
             python mock_rtl_tcp.py --wav speech.wav      # real WAV file
Terminal 2:  python sdr_decoder.py --freq 99.5
"""

import argparse
import array
import math
import select
import socket
import struct
import sys
import threading
import time
import wave

# ---------------------------------------------------------------------------
# Server parameters  (must match sdr_decoder.py constants)
# ---------------------------------------------------------------------------
HOST          = '127.0.0.1'
PORT          = 1234
SDR_RATE_FM   = 1_152_000    # for WBFM mode  (24 × 48 kHz)
SDR_RATE_AM   = 240_000      # for AM/SSB modes (5 × 48 kHz)
AUDIO_RATE    = 48_000       # audio sample rate inside the modulator
FREQ_DEV      = 75_000       # FM frequency deviation (Hz) – WBFM standard
SEND_CHUNK    = 8192         # bytes sent per sendall() call
STATUS_EVERY  = 5.0          # seconds between throughput log lines

# Dongle magic header sent on connect:  b'RTL0' | tuner_type=R820T | gain_steps=29
MAGIC = struct.pack('>4sII', b'RTL0', 5, 29)

COMMAND_NAMES = {
    1: 'SET_FREQ',  2: 'SET_RATE',  3: 'SET_GAIN_MODE',
    4: 'SET_GAIN',  5: 'SET_PPM',   6: 'SET_IF_GAIN',  8: 'SET_AGC',
}


# ---------------------------------------------------------------------------
# Audio generators
# ---------------------------------------------------------------------------

def _gen_synthetic_voice(n_samples: int) -> list:
    """Return `n_samples` of a synthetic voice-like waveform in [-1.0, +1.0].

    Simulates a voiced vowel: fundamental F0 ≈ 150 Hz with harmonics and two
    formant resonances, amplitude-modulated at a ~3 Hz syllable rate.
    It sounds like a steady hum/buzz — clearly not noise, not speech, but
    recognisably voice-like.  For real speech use --wav.
    """
    audio = []
    inv_rate = 1.0 / AUDIO_RATE
    for n in range(n_samples):
        t = n * inv_rate
        # Syllable amplitude envelope: smooth raised cosine at 3 Hz
        envelope = 0.5 - 0.5 * math.cos(2 * math.pi * 3.0 * t)

        # Vocal source: harmonic stack at F0 = 150 Hz
        f0 = 150.0
        s = (
            1.00 * math.sin(2 * math.pi * f0        * t)  # fundamental
          + 0.55 * math.sin(2 * math.pi * 2 * f0    * t)  # 2nd harmonic
          + 0.30 * math.sin(2 * math.pi * 3 * f0    * t)  # 3rd harmonic
          + 0.20 * math.sin(2 * math.pi * 4 * f0    * t)  # 4th harmonic
          # Formant peaks boost harmonics near F1=700 Hz, F2=1200 Hz
          + 0.40 * math.sin(2 * math.pi * 700        * t)
          + 0.25 * math.sin(2 * math.pi * 1200       * t)
          + 0.10 * math.sin(2 * math.pi * 2500       * t)
        )
        # Normalise peak to ~1.0 and apply envelope
        s = (s / 1.80) * envelope * 0.85
        audio.append(max(-1.0, min(1.0, s)))
    return audio


def _gen_tone(freq_hz: float, n_samples: int) -> list:
    """Return `n_samples` of a pure sine tone at `freq_hz` Hz."""
    inv_rate = 1.0 / AUDIO_RATE
    return [0.85 * math.sin(2 * math.pi * freq_hz * n * inv_rate)
            for n in range(n_samples)]


def _load_wav(path: str) -> list:
    """Load a WAV file; return samples as floats in [-1.0, +1.0].

    Supports (via manual RIFF chunk parsing – Python's wave module only
    handles format 1):
      - Format 1  PCM       : 16-bit or 32-bit integer
      - Format 3  IEEE_FLOAT: 32-bit float  (common DAW / Windows export)
      - Format 65534 EXTENSIBLE: dispatches to format 1 or 3 sub-format
    """
    with open(path, 'rb') as f:
        raw = f.read()

    # ── RIFF header ──────────────────────────────────────────────────────────
    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise ValueError("Not a valid WAV/RIFF file")

    # ── Walk chunks to find 'fmt ' and 'data' ────────────────────────────────
    fmt_chunk  = None
    data_chunk = None
    pos = 12
    while pos + 8 <= len(raw):
        cid  = raw[pos:pos + 4]
        csz  = struct.unpack_from('<I', raw, pos + 4)[0]
        body = raw[pos + 8: pos + 8 + csz]
        if cid == b'fmt ':
            fmt_chunk = body
        elif cid == b'data':
            data_chunk = body
        pos += 8 + csz + (csz & 1)   # RIFF pads odd-length chunks with 1 byte
        if fmt_chunk is not None and data_chunk is not None:
            break

    if fmt_chunk is None or data_chunk is None:
        raise ValueError("WAV file is missing 'fmt ' or 'data' chunk")

    # ── Parse fmt chunk ───────────────────────────────────────────────────────
    audio_fmt       = struct.unpack_from('<H', fmt_chunk, 0)[0]
    n_channels      = struct.unpack_from('<H', fmt_chunk, 2)[0]
    sample_rate     = struct.unpack_from('<I', fmt_chunk, 4)[0]
    bits_per_sample = struct.unpack_from('<H', fmt_chunk, 14)[0]

    # EXTENSIBLE (65534): real format is in sub-format GUID first 2 bytes
    if audio_fmt == 65534 and len(fmt_chunk) >= 26:
        audio_fmt = struct.unpack_from('<H', fmt_chunk, 24)[0]

    # ── Decode samples ────────────────────────────────────────────────────────
    if audio_fmt == 1:                          # Integer PCM
        if bits_per_sample == 16:
            samples = array.array('h', data_chunk)   # signed int16
            scale   = 1.0 / 32768.0
        elif bits_per_sample == 32:
            samples = array.array('i', data_chunk)   # signed int32
            scale   = 1.0 / 2_147_483_648.0
        else:
            raise ValueError(f"Unsupported PCM bit depth: {bits_per_sample}")
        audio = [s * scale for s in samples]

    elif audio_fmt == 3:                        # IEEE 754 float
        if bits_per_sample != 32:
            raise ValueError(f"Unsupported float bit depth: {bits_per_sample}")
        samples = array.array('f', data_chunk)  # float32
        audio   = list(samples)

    else:
        raise ValueError(
            f"Unsupported WAV format tag: {audio_fmt} "
            f"(supported: 1=PCM, 3=IEEE_FLOAT, 65534=EXTENSIBLE)"
        )

    # ── Multi-channel → mono (average all channels) ───────────────────────────
    if n_channels > 1:
        audio = [
            sum(audio[i + c] for c in range(n_channels)) / n_channels
            for i in range(0, len(audio), n_channels)
        ]

    # Clamp to [-1.0, +1.0] (float WAVs may legally exceed ±1.0)
    audio = [max(-1.0, min(1.0, s)) for s in audio]

    # ── Resample to AUDIO_RATE if needed (nearest-neighbour) ─────────────────
    if sample_rate != AUDIO_RATE:
        ratio = AUDIO_RATE / sample_rate
        n_out = int(len(audio) * ratio)
        audio = [audio[min(int(i / ratio), len(audio) - 1)] for i in range(n_out)]
        print(f"[mock] Resampled {sample_rate} Hz → {AUDIO_RATE} Hz")

    fmt_name = {1: 'PCM-int', 3: 'IEEE-float'}.get(audio_fmt, f'fmt={audio_fmt}')
    print(f"[mock] WAV loaded: {len(audio)/AUDIO_RATE:.1f}s  "
          f"{fmt_name} {bits_per_sample}-bit  {n_channels}ch  "
          f"peak={max(abs(s) for s in audio):.3f}")
    return audio


# ---------------------------------------------------------------------------
# Modulators
# ---------------------------------------------------------------------------

def _fm_modulate(audio: list, sdr_rate: int, freq_dev: float,
                 phase_start: float = 0.0) -> tuple[bytes, float]:
    """FM-modulate `audio` samples into interleaved uint8 I/Q bytes."""
    upsample    = sdr_rate // AUDIO_RATE
    n_iq        = len(audio) * upsample
    buf         = bytearray(n_iq * 2)
    phase       = phase_start
    phase_scale = 2.0 * math.pi * freq_dev / sdr_rate
    idx         = 0

    for s in audio:
        step = phase_scale * s
        for _ in range(upsample):
            phase += step
            buf[idx]     = int(127.5 + 127.5 * math.cos(phase)) & 0xFF
            buf[idx + 1] = int(127.5 + 127.5 * math.sin(phase)) & 0xFF
            idx += 2

    phase = (phase + math.pi) % (2 * math.pi) - math.pi
    return bytes(buf), phase


def _am_modulate(audio: list, sdr_rate: int,
                 carrier_hz: float = 0.0,
                 suppress_carrier: bool = False) -> tuple[bytes, float]:
    """AM-modulate `audio` samples into interleaved uint8 I/Q bytes.

    When suppress_carrier=False (normal AM):
        I = 127.5 + 60 * (1 + m * audio)  — carrier + two sidebands
    When suppress_carrier=True (DSB-SC, for SSB testing):
        I = 127.5 + 60 * audio            — sidebands only, no carrier
    """
    upsample  = sdr_rate // AUDIO_RATE
    n_iq      = len(audio) * upsample
    buf       = bytearray(n_iq * 2)
    phase     = 0.0
    p_step    = 2.0 * math.pi * carrier_hz / sdr_rate
    idx       = 0
    mod_depth = 0.85

    for s in audio:
        clamped = max(-1.0, min(1.0, s))
        if suppress_carrier:
            env = mod_depth * clamped          # no DC offset → no carrier
        else:
            env = 1.0 + mod_depth * clamped    # DC offset = carrier
        for _ in range(upsample):
            if carrier_hz == 0.0:
                buf[idx]     = int(127.5 + 110.0 * env) & 0xFF
                buf[idx + 1] = 128
            else:
                buf[idx]     = int(127.5 + 110.0 * env * math.cos(phase)) & 0xFF
                buf[idx + 1] = int(127.5 + 110.0 * env * math.sin(phase)) & 0xFF
                phase += p_step
            idx += 2

    phase = (phase + math.pi) % (2 * math.pi) - math.pi
    return bytes(buf), phase


def _hilbert_fir(signal: list, n_taps: int = 63) -> list:
    """Apply FIR Hilbert transform (90° phase shift) to a signal.

    Uses a windowed-sinc Hilbert kernel:
      h[n] = (2 / (π·n)) × window[n]   for odd n
      h[n] = 0                          for even n
    where window is a Hamming window.  Returns the Hilbert-transformed signal
    (same length as input).
    """
    half = n_taps // 2
    # Build kernel
    kernel = [0.0] * n_taps
    for i in range(n_taps):
        n = i - half
        if n % 2 != 0:
            kernel[i] = 2.0 / (math.pi * n)
    # Apply Hamming window
    for i in range(n_taps):
        kernel[i] *= 0.54 - 0.46 * math.cos(2.0 * math.pi * i / (n_taps - 1))

    # Convolve (direct form, adequate for offline pre-rendering)
    out = [0.0] * len(signal)
    sig_len = len(signal)
    for i in range(sig_len):
        acc = 0.0
        for j in range(n_taps):
            src = i - j + half
            if 0 <= src < sig_len:
                acc += kernel[j] * signal[src]
        out[i] = acc
    return out


def _ssb_modulate(audio: list, sdr_rate: int,
                  sideband: str = 'lsb') -> tuple[bytes, float]:
    """Generate true single-sideband I/Q using the Hilbert/phasing method.

    Analytic signal: a(t) = audio(t) + j·hilbert(audio(t))
      - USB: I/Q = a(t)        → content at positive frequencies only
      - LSB: I/Q = conj(a(t))  → content at negative frequencies only

    The decoder's BFO then shifts the content to baseband.
    """
    print("[hilbert] ", end='', flush=True)
    audio_h = _hilbert_fir(audio)

    upsample = sdr_rate // AUDIO_RATE
    n_iq     = len(audio) * upsample
    buf      = bytearray(n_iq * 2)
    idx      = 0
    scale    = 110.0

    sign_q = -1.0 if sideband == 'lsb' else 1.0

    for k in range(len(audio)):
        i_val = max(-1.0, min(1.0, audio[k]))
        q_val = max(-1.0, min(1.0, audio_h[k])) * sign_q
        i_byte = int(127.5 + scale * i_val) & 0xFF
        q_byte = int(127.5 + scale * q_val) & 0xFF
        for _ in range(upsample):
            buf[idx]     = i_byte
            buf[idx + 1] = q_byte
            idx += 2

    return bytes(buf), 0.0


def _build_iq_loop(audio: list, sdr_rate: int,
                   mod: str, freq_dev: float,
                   sideband: str = '') -> tuple[memoryview, float]:
    """Pre-render the entire audio list into an IQ byte loop buffer."""
    upsample = sdr_rate // AUDIO_RATE
    print(f"[mock] Pre-rendering {len(audio)/AUDIO_RATE:.1f}s of {mod.upper()} I/Q "
          f"({len(audio) * upsample * 2 / 1e6:.1f} MB)  … ", end='', flush=True)
    t0 = time.monotonic()
    if mod == 'fm':
        iq_bytes, final_phase = _fm_modulate(audio, sdr_rate, freq_dev)
    elif mod == 'ssb':
        iq_bytes, final_phase = _ssb_modulate(audio, sdr_rate, sideband)
    else:
        iq_bytes, final_phase = _am_modulate(audio, sdr_rate)
    print(f"done in {time.monotonic()-t0:.1f}s")
    return memoryview(iq_bytes), final_phase


# ---------------------------------------------------------------------------
# Command drain thread
# ---------------------------------------------------------------------------

def _drain_commands(conn: socket.socket, stop: threading.Event) -> None:
    """Read and log 5-byte command packets from the decoder client.

    Uses select() so settimeout() is never called on the shared socket
    (which would interfere with sendall() in the send thread).
    """
    buf = bytearray()
    while not stop.is_set():
        ready, _, _ = select.select([conn], [], [], 0.05)
        if not ready:
            continue
        try:
            chunk = conn.recv(256)
        except OSError:
            break
        if not chunk:
            break
        buf.extend(chunk)
        while len(buf) >= 5:
            cmd_id, value = struct.unpack('>BI', buf[:5])
            name = COMMAND_NAMES.get(cmd_id, f'CMD_{cmd_id}')
            if cmd_id == 1:
                print(f"  [mock cmd] {name:<16} value={value}  "
                      f"({value/1e6:.3f} MHz)")
            else:
                print(f"  [mock cmd] {name:<16} value={value}")
            del buf[:5]


# ---------------------------------------------------------------------------
# Client handler
# ---------------------------------------------------------------------------

def handle_client(conn: socket.socket, addr: tuple,
                  loop_buf: memoryview, loop_phase: float,
                  bytes_per_sec: int) -> None:
    """Serve one connected client: stream I/Q paced to the configured SDR rate."""
    print(f"[mock] Client connected from {addr[0]}:{addr[1]}")

    # Send magic header
    try:
        conn.sendall(MAGIC)
    except OSError as exc:
        print(f"[mock] Failed to send magic: {exc}")
        conn.close()
        return

    # Start command drain thread
    stop_evt   = threading.Event()
    cmd_thread = threading.Thread(
        target=_drain_commands, args=(conn, stop_evt), daemon=True
    )
    cmd_thread.start()

    # Stream paced I/Q bytes
    buf_len    = len(loop_buf)
    bps        = bytes_per_sec   # local alias for tight loop
    bytes_sent = 0
    t0         = time.monotonic()
    last_print = t0
    offset     = 0

    try:
        while True:
            # Pacing: sleep until the next chunk is due
            elapsed = time.monotonic() - t0
            quota   = int(elapsed * bps)
            behind  = quota - bytes_sent
            if behind <= 0:
                sleep_s = (bytes_sent + SEND_CHUNK - quota) / bps
                time.sleep(max(sleep_s, 0.001))
                continue

            # Slice from the pre-rendered loop buffer
            end = offset + SEND_CHUNK
            if end <= buf_len:
                chunk  = loop_buf[offset:end]   # zero-copy memoryview slice
                offset = end % buf_len
            else:
                chunk  = bytes(loop_buf[offset:]) + bytes(loop_buf[:end - buf_len])
                offset = end - buf_len

            conn.sendall(chunk)
            bytes_sent += SEND_CHUNK

            # Periodic throughput log
            now = time.monotonic()
            if now - last_print >= STATUS_EVERY:
                rate = bytes_sent / (now - t0) / 1e6
                print(f"[mock] Sent {bytes_sent/1e6:.1f} MB  "
                      f"rate={rate:.2f} MB/s  elapsed={now-t0:.0f}s")
                last_print = now

    except OSError:
        pass  # client disconnected
    finally:
        stop_evt.set()
        cmd_thread.join(timeout=0.2)
        conn.close()
        print(f"[mock] Client {addr[0]}:{addr[1]} disconnected  "
              f"({bytes_sent/1e6:.2f} MB sent)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mock RTL-TCP server with FM/AM modulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python mock_rtl_tcp.py                              # WBFM synthetic voice\n"
            "  python mock_rtl_tcp.py --wav speech.wav              # WBFM from WAV\n"
            "  python mock_rtl_tcp.py --mod am --tone 1000          # AM test tone\n"
            "  python mock_rtl_tcp.py --mod am --wav speech.wav     # AM speech\n"
        ),
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--wav',  metavar='FILE', help="Modulate a WAV file (16/32-bit PCM or float)")
    grp.add_argument('--tone', metavar='FREQ', type=float, default=None,
                     help="Emit a pure sine tone at FREQ Hz")
    parser.add_argument('--mod', choices=['fm', 'am', 'lsb', 'usb'], default='fm',
                        help="Modulation type: fm, am, lsb, or usb")
    parser.add_argument('--dev', type=float, default=FREQ_DEV,
                        help="FM frequency deviation in Hz (FM only)")
    parser.add_argument('--sdr-rate', type=int, default=None,
                        help="Override SDR sample rate (default: 1152000 for FM, 240000 for AM/SSB)")
    args = parser.parse_args()

    freq_dev = args.dev
    mod      = args.mod

    # Determine internal modulator type and sideband
    if mod == 'fm':
        mod_type = 'fm'
        sideband = ''
    elif mod in ('lsb', 'usb'):
        mod_type = 'ssb'
        sideband = mod
    else:
        mod_type = 'am'
        sideband = ''

    # Pick SDR rate based on modulation mode (or user override)
    if args.sdr_rate is not None:
        sdr_rate = args.sdr_rate
    elif mod == 'fm':
        sdr_rate = SDR_RATE_FM
    else:
        sdr_rate = SDR_RATE_AM

    bytes_per_sec = sdr_rate * 2

    # Build audio source ─────────────────────────────────────────────────────
    if args.wav:
        try:
            audio = _load_wav(args.wav)
        except Exception as exc:
            print(f"[mock] ERROR loading WAV: {exc}")
            sys.exit(1)
        source_desc = f"WAV file: {args.wav}"
    elif args.tone is not None:
        audio = _gen_tone(args.tone, AUDIO_RATE * 5)
        source_desc = f"Pure tone: {args.tone:.1f} Hz"
    else:
        audio = _gen_synthetic_voice(AUDIO_RATE * 4)
        source_desc = "Synthetic voice (use --wav for real speech)"

    # Pre-render I/Q ─────────────────────────────────────────────────────────
    loop_buf, loop_phase = _build_iq_loop(audio, sdr_rate, mod_type, freq_dev,
                                           sideband=sideband)

    # Start TCP server ───────────────────────────────────────────────────────
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        srv.bind((HOST, PORT))
    except OSError as exc:
        print(f"[mock] Cannot bind {HOST}:{PORT}: {exc}")
        print("[mock] Is another rtl_tcp (real or mock) already running?")
        sys.exit(1)

    srv.listen(4)
    print(f"\n[mock] RTL-TCP mock server ready on {HOST}:{PORT}")
    print(f"[mock] Source  : {source_desc}")
    print(f"[mock] Mod     : {mod.upper()}  |  SDR rate: {sdr_rate/1e6:.3f} MSPS")
    if mod == 'fm':
        print(f"[mock] FM dev  : {freq_dev/1e3:.0f} kHz")
    print(f"[mock] Loop    : {len(audio)/AUDIO_RATE:.1f}s")
    print("[mock] Press Ctrl+C to stop.\n")

    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, loop_buf, loop_phase, bytes_per_sec),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        print("\n[mock] Shutting down.")
    finally:
        srv.close()


if __name__ == '__main__':
    main()
