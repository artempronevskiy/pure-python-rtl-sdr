# -*- coding: utf-8 -*-
"""
Pure Python RTL-SDR Wideband FM Decoder
Requires: Python 3.10+, stdlib only, rtl_tcp running on localhost:1234

Usage:
    rtl_tcp                              <- terminal 1 (start the dongle server)
    python sdr_decoder.py --freq 98.5   <- terminal 2 (tune and decode)
"""

import argparse
import ctypes
import ctypes.wintypes as wintypes
import math
import socket
import struct
import sys
import time
from typing import Any


# =============================================================================
# Windows Multimedia API – structure definitions
# =============================================================================

WAVE_FORMAT_PCM = 1
WHDR_DONE       = 0x00000001
WAVE_MAPPER     = 0xFFFFFFFF   # bare Python int, not ctypes wrapper


class WAVEFORMATEX(ctypes.Structure):
    """waveformatex structure – describes the PCM stream format."""
    _fields_ = [
        ('wFormatTag',      wintypes.WORD),
        ('nChannels',       wintypes.WORD),
        ('nSamplesPerSec',  wintypes.DWORD),
        ('nAvgBytesPerSec', wintypes.DWORD),
        ('nBlockAlign',     wintypes.WORD),
        ('wBitsPerSample',  wintypes.WORD),
        ('cbSize',          wintypes.WORD),
    ]


class WAVEHDR(ctypes.Structure):
    """wavehdr structure – describes a waveform-audio buffer submitted to the driver.

    Windows SDK field types:
        lpData          LPSTR        → pointer-sized char pointer
        dwBufferLength  DWORD        → 32-bit
        dwBytesRecorded DWORD        → 32-bit
        dwUser          DWORD_PTR    → pointer-sized (64-bit on Win64)
        dwFlags         DWORD        → 32-bit
        dwLoops         DWORD        → 32-bit
        lpNext          struct*      → pointer-sized
        reserved        DWORD_PTR    → pointer-sized
    """


# Deferred assignment required for the self-referential lpNext pointer
WAVEHDR._fields_ = [
    ('lpData',          ctypes.c_void_p),       # LPSTR – raw pointer
    ('dwBufferLength',  wintypes.DWORD),
    ('dwBytesRecorded', wintypes.DWORD),
    ('dwUser',          ctypes.c_size_t),        # DWORD_PTR
    ('dwFlags',         wintypes.DWORD),
    ('dwLoops',         wintypes.DWORD),
    ('lpNext',          ctypes.POINTER(WAVEHDR)),
    ('reserved',        ctypes.c_size_t),        # DWORD_PTR
]


def _setup_winmm() -> Any:
    """Load winmm.dll and configure argtypes/restype for all used functions.

    Setting restype is critical: MMRESULT is UINT (unsigned).  ctypes defaults
    to c_int (signed), which can misinterpret large error codes.

    This function is intentionally called only once at module level and the
    configured DLL object is reused by all WinAudioOut instances.  Re-calling
    it would redundantly overwrite global argtypes on the shared DLL object.
    """
    mm = ctypes.windll.winmm

    _UINT      = ctypes.c_uint
    _MMRESULT  = ctypes.c_uint
    _HWAVEOUT  = ctypes.c_void_p
    _PHWAVEOUT = ctypes.POINTER(ctypes.c_void_p)
    _PWFX      = ctypes.POINTER(WAVEFORMATEX)
    _PWAVEHDR  = ctypes.POINTER(WAVEHDR)
    _UINT_PTR  = ctypes.c_size_t
    _DWORD     = wintypes.DWORD

    mm.waveOutOpen.restype          = _MMRESULT
    mm.waveOutOpen.argtypes         = [_PHWAVEOUT, _UINT, _PWFX,
                                        _UINT_PTR, _UINT_PTR, _DWORD]

    mm.waveOutPrepareHeader.restype  = _MMRESULT
    mm.waveOutPrepareHeader.argtypes = [_HWAVEOUT, _PWAVEHDR, _UINT]

    mm.waveOutWrite.restype          = _MMRESULT
    mm.waveOutWrite.argtypes         = [_HWAVEOUT, _PWAVEHDR, _UINT]

    mm.waveOutReset.restype          = _MMRESULT
    mm.waveOutReset.argtypes         = [_HWAVEOUT]

    mm.waveOutUnprepareHeader.restype  = _MMRESULT
    mm.waveOutUnprepareHeader.argtypes = [_HWAVEOUT, _PWAVEHDR, _UINT]

    mm.waveOutClose.restype          = _MMRESULT
    mm.waveOutClose.argtypes         = [_HWAVEOUT]

    return mm


# Module-level singleton: configured once, shared by all WinAudioOut instances.
_WINMM: Any = _setup_winmm()


# =============================================================================
# Windows Audio Sink
# =============================================================================

class WinAudioOut:
    """Real-time PCM output via Windows Multimedia API (winmm.dll).

    Uses multi-buffering so the DSP loop is never stalled waiting for the
    hardware to drain the previous chunk.

    Buffer lifetime contract
    ------------------------
    lpData in WAVEHDR points directly into the bytes object held in
    self._bufs[i].  The bytes reference must stay alive until the driver
    marks the header WHDR_DONE and waveOutUnprepareHeader has been called.
    We guarantee this by storing the bytes in self._bufs[i] before the
    waveOutPrepareHeader call and only clearing it after UnprepareHeader.
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 1,
                 bits_per_sample: int = 16, num_buffers: int = 4) -> None:
        self._mm         = _WINMM   # use the pre-configured module-level singleton
        self._hwo        = ctypes.c_void_p(0)  # HWAVEOUT
        self._opened     = False
        self.num_buffers = num_buffers

        wfx = WAVEFORMATEX()
        wfx.wFormatTag      = WAVE_FORMAT_PCM
        wfx.nChannels       = channels
        wfx.nSamplesPerSec  = sample_rate
        wfx.wBitsPerSample  = bits_per_sample
        wfx.nBlockAlign     = (channels * bits_per_sample) // 8
        wfx.nAvgBytesPerSec = sample_rate * wfx.nBlockAlign
        wfx.cbSize          = 0

        res = self._mm.waveOutOpen(
            ctypes.byref(self._hwo),    # receives the HWAVEOUT handle
            WAVE_MAPPER,                # default output device (plain int)
            ctypes.byref(wfx),
            0, 0, 0,
        )
        if res != 0:
            raise RuntimeError(f"waveOutOpen failed: MMRESULT={res:#010x}")

        self._opened = True
        # _bufs stores ctypes c_char array objects (from create_string_buffer),
        # not plain bytes – typed as Any because ctypes buffer types are opaque.
        self._bufs: list[Any]            = [None] * num_buffers
        self._hdrs: list[WAVEHDR | None] = [None] * num_buffers

    # ------------------------------------------------------------------
    def play_chunk(self, data: bytes) -> None:
        """Submit a PCM buffer to the hardware.

        Returns as soon as a free slot is found.  If all slots are in-flight
        the call sleeps briefly (back-pressure) until one becomes free.
        """
        while True:
            for i in range(self.num_buffers):
                hdr = self._hdrs[i]
                if hdr is None or (hdr.dwFlags & WHDR_DONE):
                    # Release the previously prepared header and its buffer
                    if hdr is not None:
                        self._mm.waveOutUnprepareHeader(
                            self._hwo,
                            ctypes.byref(hdr),
                            ctypes.sizeof(WAVEHDR),
                        )
                        self._hdrs[i] = None
                        self._bufs[i] = None  # release ctypes buffer reference

                    # Wrap data in a ctypes buffer so we hold a stable C pointer.
                    # Using create_string_buffer (a ctypes c_char array) gives us:
                    #   - A fixed C-level memory address (unlike raw bytes which
                    #     Python may move or GC while the driver is reading it).
                    #   - A type that ctypes.cast accepts (plain bytes does not).
                    # We store the buffer in self._bufs[i] to keep it alive until
                    # after waveOutUnprepareHeader is called for this slot.
                    cbuf = ctypes.create_string_buffer(data)
                    self._bufs[i] = cbuf

                    new_hdr = WAVEHDR()
                    new_hdr.lpData         = ctypes.cast(cbuf, ctypes.c_void_p)
                    new_hdr.dwBufferLength = len(data)
                    new_hdr.dwFlags        = 0
                    new_hdr.dwLoops        = 0

                    res = self._mm.waveOutPrepareHeader(
                        self._hwo,
                        ctypes.byref(new_hdr),
                        ctypes.sizeof(WAVEHDR),
                    )
                    if res != 0:
                        raise RuntimeError(f"waveOutPrepareHeader: MMRESULT={res:#010x}")

                    res = self._mm.waveOutWrite(
                        self._hwo,
                        ctypes.byref(new_hdr),
                        ctypes.sizeof(WAVEHDR),
                    )
                    if res != 0:
                        raise RuntimeError(f"waveOutWrite: MMRESULT={res:#010x}")

                    self._hdrs[i] = new_hdr
                    return  # submitted – return control to DSP loop

            # All slots busy: yield briefly to avoid busy-spinning at 100% CPU
            time.sleep(0.005)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Drain pending buffers and release the hardware device."""
        if not self._opened:
            return

        # waveOutReset marks all in-flight headers WHDR_DONE immediately.
        # Without this, waveOutClose would return MMSYSERR_STILLPLAYING.
        self._mm.waveOutReset(self._hwo)

        # Give the driver up to 2 s to return all headers.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if all(h is None or (h.dwFlags & WHDR_DONE) for h in self._hdrs):
                break
            time.sleep(0.01)

        for hdr in self._hdrs:
            if hdr is not None:
                self._mm.waveOutUnprepareHeader(
                    self._hwo,
                    ctypes.byref(hdr),
                    ctypes.sizeof(WAVEHDR),
                )
        self._bufs = [None] * self.num_buffers  # type: list[Any]

        self._mm.waveOutClose(self._hwo)
        self._opened = False


# =============================================================================
# RTL-TCP Client
# =============================================================================

class RtlTcpClient:
    """Connect to an `rtl_tcp` server and stream raw unsigned-8 I/Q samples.

    RTL-TCP wire protocol
    ---------------------
    On connect the server sends a 12-byte magic header::

        Bytes  0- 3  : b'RTL0'              (magic)
        Bytes  4- 7  : tuner type           (uint32 BE)
        Bytes  8-11  : number of gain steps (uint32 BE)

    All commands are exactly 5 bytes big-endian::

        Byte  0     : command id  (uint8)
        Bytes 1-4   : value       (uint32 BE)

    Command IDs:
        1  Set center frequency   (Hz)
        2  Set sample rate        (sps)
        3  Set tuner gain mode    0 = manual, 1 = automatic AGC
        4  Set tuner gain         tenths of dB  (e.g. 496 → 49.6 dB)
        5  Set freq correction    PPM (signed int passed as uint32)
        6  Set IF gain            high byte = stage index, low 3 bytes = gain
        8  Set RTL2832 AGC        0 = disabled, 1 = enabled
    """

    TUNER_NAMES = {
        0: 'Unknown', 1: 'E4000', 2: 'FC0012',
        3: 'FC0013',  4: 'FC2580', 5: 'R820T', 6: 'R828D',
    }

    def __init__(self, host: str = '127.0.0.1', port: int = 1234) -> None:
        self.host   = host
        self.port   = port
        self._sock: socket.socket | None = None

    # ------------------------------------------------------------------
    def connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle – we send small 5-byte commands, don't want them batched
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Large OS receive buffer to absorb bursts at >2 MB/s
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.settimeout(5.0)

        print(f"Connecting to rtl_tcp at {self.host}:{self.port} ...")
        try:
            sock.connect((self.host, self.port))
        except OSError as exc:   # covers ConnectionRefusedError, TimeoutError, etc.
            sock.close()
            print("=" * 48)
            print(f"ERROR: Cannot reach rtl_tcp ({exc})")
            print("Start it first:  rtl_tcp -a 127.0.0.1")
            print("=" * 48)
            sys.exit(1)

        self._sock = sock

        # Read 12-byte dongle magic header (still within the 5 s timeout)
        try:
            magic = self._read_exactly(12)
        except (ConnectionError, TimeoutError) as exc:
            self.close()
            raise RuntimeError(f"rtl_tcp did not send magic header: {exc}") from exc

        # Switch to blocking mode with no timeout for the streaming phase
        self._sock.settimeout(None)

        tag, tuner_type, gain_count = struct.unpack('>4sII', magic)
        tuner_name = self.TUNER_NAMES.get(tuner_type, f'type={tuner_type}')
        print(f"Dongle: '{tag.decode(errors='replace')}', "
              f"tuner={tuner_name}, gain_steps={gain_count}")

    # ------------------------------------------------------------------
    def _send_cmd(self, cmd_id: int, value: int) -> None:
        """Send a single 5-byte command to rtl_tcp."""
        if self._sock is None:
            raise RuntimeError("_send_cmd called before connect()")
        # value is treated as uint32 on the wire; mask prevents struct error
        # on negative PPM corrections etc.
        self._sock.sendall(struct.pack('>BI', cmd_id, value & 0xFFFF_FFFF))

    def set_center_freq(self, freq_hz: int) -> None:
        """Command 1: Set tuner center frequency (Hz)."""
        self._send_cmd(1, int(freq_hz))

    def set_sample_rate(self, rate_hz: int) -> None:
        """Command 2: Set ADC sample rate (samples/s)."""
        self._send_cmd(2, int(rate_hz))

    def set_tuner_gain_mode(self, manual: bool) -> None:
        """Command 3: False = automatic AGC, True = manual gain via set_gain()."""
        self._send_cmd(3, 0 if manual else 1)

    def set_gain(self, gain_tenths_db: int) -> None:
        """Command 4: Tuner gain in tenths of dB (e.g. 496 → 49.6 dB)."""
        self._send_cmd(4, int(gain_tenths_db))

    def set_freq_correction(self, ppm: int) -> None:
        """Command 5: Frequency correction in PPM (can be negative)."""
        self._send_cmd(5, int(ppm))

    def set_rtl_agc(self, enabled: bool) -> None:
        """Command 8: RTL2832 internal digital AGC (separate from tuner AGC)."""
        self._send_cmd(8, 1 if enabled else 0)

    def set_direct_sampling(self, mode: int) -> None:
        """Command 9: Direct sampling mode.

        0 = disabled (normal quadrature), 1 = I-ADC input, 2 = Q-ADC input.
        Required for LW/MW reception below ~24 MHz on RTL2832U dongles with
        direct sampling hardware mod.
        """
        self._send_cmd(9, int(mode))

    # ------------------------------------------------------------------
    def read_samples(self, num_bytes: int) -> bytes:
        """Read exactly num_bytes of interleaved uint8 I/Q data."""
        return self._read_exactly(num_bytes)

    def _read_exactly(self, num_bytes: int) -> bytes:
        buf      = bytearray(num_bytes)
        view     = memoryview(buf)
        received = 0
        while received < num_bytes:
            n = self._sock.recv_into(view[received:], num_bytes - received)
            if n == 0:
                raise ConnectionError("rtl_tcp closed the connection unexpectedly.")
            received += n
        return bytes(buf)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# =============================================================================
# Band / mode configuration
# =============================================================================

# Each band preset sets: default freq (MHz), tuning step (kHz),
# demod mode, SDR sample rate, default audio gain, de-emphasis (µs or None).
BANDS: dict[str, dict] = {
    'fm':  {'freq': 99.5,    'step': 100.0,  'mode': 'wbfm', 'sdr_rate': 1_152_000,
            'gain': 6000.0,  'deemph': 75.0,  'desc': 'FM broadcast (87.5–108 MHz)'},
    'lw':  {'freq': 0.198,   'step': 9.0,    'mode': 'am',   'sdr_rate': 240_000,
            'gain': 50000.0, 'deemph': None,   'desc': 'Long wave (148–283 kHz)'},
    'mw':  {'freq': 1.000,   'step': 9.0,    'mode': 'am',   'sdr_rate': 240_000,
            'gain': 50000.0, 'deemph': None,   'desc': 'Medium wave (520–1710 kHz)'},
    'sw':  {'freq': 7.200,   'step': 5.0,    'mode': 'am',   'sdr_rate': 240_000,
            'gain': 50000.0, 'deemph': None,   'desc': 'Short wave (1.7–30 MHz)'},
    'air': {'freq': 121.500, 'step': 25.0,   'mode': 'am',   'sdr_rate': 240_000,
            'gain': 50000.0, 'deemph': None,   'desc': 'Airband AM (108–137 MHz)'},
    'ham': {'freq': 7.100,   'step': 0.1,    'mode': 'lsb',  'sdr_rate': 240_000,
            'gain': 80000.0, 'deemph': None,   'desc': 'Amateur radio SSB'},
}

MODE_NAMES = {'wbfm': 'WBFM', 'am': 'AM', 'lsb': 'LSB', 'usb': 'USB'}

AUDIO_RATE = 48_000   # Hz – standard audio output sample rate


# =============================================================================
# DSP pipeline – demodulators
# =============================================================================

_TWO_PI = math.tau
_PI     = math.pi


def _pack_pcm(samples: list, gain: float, state: dict,
              use_deemph: bool = False, alpha: float = 0.0) -> bytes:
    """Convert float samples to little-endian int16 PCM bytes.

    Optionally applies 1-pole IIR de-emphasis (for WBFM).
    Applies gain scaling, hard clipping to int16 range.
    """
    prev_d  = state.get('prev_demph', 0.0)
    pcm_buf = bytearray(len(samples) * 2)
    idx     = 0
    for s in samples:
        if use_deemph:
            prev_d += alpha * (s - prev_d)
            v = int(prev_d * gain)
        else:
            v = int(s * gain)
        if   v >  32767: v =  32767
        elif v < -32768: v = -32768
        pcm_buf[idx]     = v & 0xFF
        pcm_buf[idx + 1] = (v >> 8) & 0xFF
        idx += 2
    state['prev_demph'] = prev_d
    return bytes(pcm_buf)


# ── WBFM demodulator ────────────────────────────────────────────────────────

def demod_wbfm(raw: bytes, state: dict, gain: float, alpha: float) -> bytes:
    """Wideband FM demodulation.  SDR_RATE = 1 152 000 sps.

    Chain: ×4 decimate → phase discriminator → ×6 decimate → de-emphasis → PCM
    """
    # 1. Decimate ×4 (subsample every 4th I/Q pair)
    I_dec = raw[0::8]
    Q_dec = raw[1::8]

    # 2. FM discriminator: atan2(Q,I) diff
    angles = [math.atan2(q - 127.5, i - 127.5) for i, q in zip(I_dec, Q_dec)]
    prev_a = state['prev_angle']
    diffs = []
    for a in angles:
        d = a - prev_a
        if   d >  _PI: d -= _TWO_PI
        elif d < -_PI: d += _TWO_PI
        diffs.append(d)
        prev_a = a
    state['prev_angle'] = prev_a

    # 3. Box-car decimate ×6 (288 kSPS → 48 kSPS)
    n = len(diffs)
    n6 = n - (n % 6)
    inv6 = 1.0 / 6.0
    audio = [
        (diffs[i] + diffs[i+1] + diffs[i+2] +
         diffs[i+3] + diffs[i+4] + diffs[i+5]) * inv6
        for i in range(0, n6, 6)
    ]

    # 4. De-emphasis + gain + PCM
    return _pack_pcm(audio, gain, state, use_deemph=True, alpha=alpha)


# ── AM demodulator ──────────────────────────────────────────────────────────

def demod_am(raw: bytes, state: dict, gain: float, _alpha: float) -> bytes:
    """AM envelope detection.  SDR_RATE = 240 000 sps.

    Chain: envelope (√(I²+Q²)) at full rate → ×5 box-car decimate → DC removal → PCM
    """
    # 1. Envelope at full sample rate (no pre-decimation to avoid aliasing)
    n_pairs = len(raw) // 2
    env_full = [0.0] * n_pairs
    for k in range(n_pairs):
        i_val = raw[k * 2]     - 127.5
        q_val = raw[k * 2 + 1] - 127.5
        env_full[k] = math.sqrt(i_val * i_val + q_val * q_val)

    # 2. Decimate ×5 with box-car averaging (anti-aliasing filter)
    n5   = n_pairs - (n_pairs % 5)
    inv5 = 1.0 / 5.0
    envelope = [
        (env_full[j] + env_full[j+1] + env_full[j+2] +
         env_full[j+3] + env_full[j+4]) * inv5
        for j in range(0, n5, 5)
    ]

    # 3. DC removal: subtract running average (1-pole IIR high-pass)
    dc = state.get('am_dc', 0.0)
    alpha_dc = 0.001
    audio = []
    for e in envelope:
        dc += alpha_dc * (e - dc)
        audio.append(e - dc)
    state['am_dc'] = dc

    return _pack_pcm(audio, gain, state)


# ── SSB demodulator ─────────────────────────────────────────────────────────

def demod_ssb(raw: bytes, state: dict, gain: float, _alpha: float,
              sideband: str = 'lsb') -> bytes:
    """Single-sideband demodulation.  SDR_RATE = 240 000 sps.

    For a true SSB signal at complex baseband (generated by the Hilbert
    method in the mock, or received by a real SDR tuned to the carrier):
      - LSB content is at negative frequencies
      - USB content is at positive frequencies

    Taking the real part (I channel) of the complex signal recovers the
    audio for BOTH sidebands — negative freqs fold onto positive naturally.
    No BFO is needed.

    Chain: extract I → ×5 box-car decimate → DC block → PCM
    """
    # 1. Extract I channel at full rate
    n_pairs = len(raw) // 2
    i_full = [0.0] * n_pairs
    for k in range(n_pairs):
        i_full[k] = raw[k * 2] - 127.5

    # 2. Decimate ×5 with box-car averaging (anti-aliasing)
    n5   = n_pairs - (n_pairs % 5)
    inv5 = 1.0 / 5.0
    audio = [
        (i_full[j] + i_full[j+1] + i_full[j+2] +
         i_full[j+3] + i_full[j+4]) * inv5
        for j in range(0, n5, 5)
    ]

    # 3. DC-blocking high-pass
    dc = state.get('ssb_dc', 0.0)
    alpha_dc = 0.002
    for idx in range(len(audio)):
        dc += alpha_dc * (audio[idx] - dc)
        audio[idx] -= dc
    state['ssb_dc'] = dc

    return _pack_pcm(audio, gain, state)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure Python multi-mode RTL-SDR decoder (stdlib only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Band presets:\n"
            "  fm   FM broadcast 87.5–108 MHz      (WBFM, 100 kHz step)\n"
            "  lw   Long wave 148–283 kHz           (AM, 9 kHz step)\n"
            "  mw   Medium wave 520–1710 kHz        (AM, 9 kHz step)\n"
            "  sw   Short wave 1.7–30 MHz           (AM, 5 kHz step)\n"
            "  air  Airband 108–137 MHz             (AM, 25 kHz step)\n"
            "  ham  Amateur radio                   (LSB/USB, 100 Hz step)\n"
            "\n"
            "Examples:\n"
            "  python sdr_decoder.py --band fm\n"
            "  python sdr_decoder.py --band fm --freq 101.1\n"
            "  python sdr_decoder.py --band sw --freq 7.200\n"
            "  python sdr_decoder.py --mode usb --freq 14.200\n"
        ),
    )
    parser.add_argument('--band', choices=list(BANDS.keys()), default=None,
                        help="Band preset (sets mode, freq, step, gain)")
    parser.add_argument('--mode', choices=['wbfm', 'am', 'lsb', 'usb'], default=None,
                        help="Demodulation mode (overrides band preset)")
    parser.add_argument('--freq', type=float, default=None,
                        help="Frequency in MHz (overrides band default)")
    parser.add_argument('--step', type=float, default=None,
                        help="Tuning step in kHz (overrides band default)")
    parser.add_argument('--gain', type=float, default=None,
                        help="Audio amplitude scale factor")
    parser.add_argument('--deemph', type=float, default=None,
                        help="De-emphasis in µs: 50 (EU) or 75 (US). WBFM only.")
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help="rtl_tcp host")
    parser.add_argument('--port', type=int, default=1234,
                        help="rtl_tcp port")
    args = parser.parse_args()

    # ── Resolve band + mode + freq ──────────────────────────────────────────
    band_name = args.band or 'fm'
    band      = BANDS[band_name]

    mode      = args.mode or band['mode']
    freq_mhz  = args.freq if args.freq is not None else band['freq']
    step_khz  = args.step if args.step is not None else band['step']
    gain      = args.gain if args.gain is not None else band['gain']
    deemph_us = args.deemph if args.deemph is not None else (band['deemph'] or 75.0)

    # Validate de-emphasis (only meaningful for WBFM)
    if mode == 'wbfm' and deemph_us not in (50.0, 75.0):
        parser.error(f"--deemph must be 50 or 75, got {deemph_us}")

    # SDR rate: WBFM uses high rate, all others use 240 kSPS
    if mode == 'wbfm':
        sdr_rate = 1_152_000
    else:
        sdr_rate = 240_000

    # Chunk size: 100 ms of I/Q data
    chunk_samples = sdr_rate // 10
    chunk_size    = chunk_samples * 2

    # De-emphasis IIR coefficient (WBFM only)
    tau   = deemph_us * 1e-6
    alpha = 1.0 - math.exp(-1.0 / (AUDIO_RATE * tau))

    freq_hz = int(round(freq_mhz * 1e6))

    # ── Select demodulator ──────────────────────────────────────────────────
    if mode == 'wbfm':
        demod_fn = demod_wbfm
    elif mode == 'am':
        demod_fn = demod_am
    elif mode in ('lsb', 'usb'):
        # Wrap demod_ssb with sideband parameter baked in
        _side = mode
        def demod_fn(raw: bytes, state: dict, g: float, a: float) -> bytes:
            return demod_ssb(raw, state, g, a, sideband=_side)
    else:
        parser.error(f"Unknown mode: {mode}")

    # ── Connect to RTL-TCP ──────────────────────────────────────────────────
    tcp = RtlTcpClient(host=args.host, port=args.port)
    tcp.connect()
    tcp.set_sample_rate(sdr_rate)
    tcp.set_center_freq(freq_hz)
    tcp.set_tuner_gain_mode(manual=False)
    tcp.set_rtl_agc(enabled=True)

    # Enable direct sampling for frequencies below 24 MHz
    if freq_mhz < 24.0:
        tcp.set_direct_sampling(2)   # Q-ADC input (most common mod)
        print("Direct sampling mode enabled (Q-ADC) for sub-24 MHz reception")

    # ── Start audio output ──────────────────────────────────────────────────
    audio_out: WinAudioOut | None = None
    try:
        audio_out = WinAudioOut(
            sample_rate=AUDIO_RATE, channels=1,
            bits_per_sample=16, num_buffers=4,
        )

        state: dict = {
            'prev_angle': 0.0, 'prev_demph': 0.0,
            'primed': False, 'am_dc': 0.0, 'bfo_phase': 0.0,
        }

        mode_label = MODE_NAMES.get(mode, mode.upper())
        print(f"\nDecoding {mode_label}  freq={freq_mhz:.3f} MHz  "
              f"step={step_khz} kHz  gain={gain:.0f}")
        if mode == 'wbfm':
            print(f"De-emphasis: {deemph_us:.0f} µs")
        print(f"SDR rate: {sdr_rate/1e3:.0f} kSPS  |  Band: {band_name}")
        print("Press Ctrl+C to stop.\n")

        while True:
            raw = tcp.read_samples(chunk_size)

            # Prime phase angle on first chunk (WBFM only)
            if mode == 'wbfm' and not state['primed'] and len(raw) >= 2:
                state['prev_angle'] = math.atan2(
                    raw[1] - 127.5, raw[0] - 127.5)
                state['primed'] = True

            pcm = demod_fn(raw, state, gain, alpha)
            audio_out.play_chunk(pcm)

    except KeyboardInterrupt:
        print("\nStopping ...")
    except ConnectionError as exc:
        print(f"\nStream error: {exc}")
    finally:
        if audio_out is not None:
            audio_out.close()
        tcp.close()


if __name__ == "__main__":
    main()

