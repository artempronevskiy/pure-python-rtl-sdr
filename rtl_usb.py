#!/usr/bin/env python3
"""
rtl_usb.py  –  Pure-Python USB driver for RTL-SDR dongles (RTL2832U + R820T/R820T2).

Replaces rtl_tcp.exe completely.  Talks to the USB dongle via WinUSB
(ctypes bindings to setupapi.dll / winusb.dll / kernel32.dll) and
serves the I/Q stream over a TCP socket on port 1234, compatible with
the rtl_tcp protocol so that sdr_decoder.py works unchanged.

Requirements:
    - Windows (uses WinUSB / SetupAPI)
    - RTL-SDR dongle with WinUSB driver installed via Zadig
    - Python 3.10+ (stdlib only, no pip packages)

Usage:
    python rtl_usb.py [--freq 100000000] [--rate 2048000] [--gain 400] [--port 1234]
"""

from __future__ import annotations
import argparse
import ctypes
import ctypes.wintypes as wt
import socket
import struct
import threading
import time

# ---------------------------------------------------------------------------
# Section 1 – WinUSB / SetupAPI ctypes bindings
# ---------------------------------------------------------------------------

setupapi = ctypes.windll.setupapi
kernel32 = ctypes.windll.kernel32
winusb   = ctypes.windll.winusb

GENERIC_WRITE       = 0x40000000
GENERIC_READ        = 0x80000000
FILE_SHARE_READ     = 1
FILE_SHARE_WRITE    = 2
OPEN_EXISTING       = 3
FILE_ATTRIBUTE_NORMAL  = 0x80
DIGCF_PRESENT          = 0x02
DIGCF_DEVICEINTERFACE  = 0x10
ERROR_NO_MORE_ITEMS    = 259

PIPE_BULK_IN = 0x81  # RTL2832U bulk-in endpoint for I/Q data

# WinUsb_SetPipePolicy policy types
PIPE_TRANSFER_TIMEOUT = 3     # ULONG ms
RAW_IO                = 7     # BOOL


class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
    _fields_ = [
        ("cbSize",              wt.DWORD),
        ("InterfaceClassGuid",  ctypes.c_byte * 16),
        ("Flags",               wt.DWORD),
        ("Reserved",            ctypes.POINTER(ctypes.c_ulong)),
    ]


def _pack_setup_packet(request_type: int, request: int,
                       value: int, index: int, length: int) -> ctypes.c_uint64:
    """Pack a WINUSB_SETUP_PACKET into a c_uint64 for pass-by-value.

    The C API declares WinUsb_ControlTransfer(…, WINUSB_SETUP_PACKET SetupPacket, …)
    with the 8-byte struct passed BY VALUE.  ctypes normally passes Structure
    objects by reference (pointer), which would corrupt the call on 32-bit Python.
    Packing the 8 bytes into a c_uint64 forces ctypes to push them onto the stack
    as a single 64-bit value — correct on both x86 and x64.
    """
    raw = struct.pack('<BBHHH',
                      request_type & 0xFF,
                      request & 0xFF,
                      value & 0xFFFF,
                      index & 0xFFFF,
                      length & 0xFFFF)
    return ctypes.c_uint64(int.from_bytes(raw, byteorder='little'))


def _find_device_path(vid: int, pid: int) -> str | None:
    """Find the WinUSB device path for a USB device by VID/PID."""
    # Generic USB device interface GUID {A5DCBF10-6530-11D2-901F-00C04FB951ED}
    GUID_DEVINTERFACE_USB_DEVICE = (ctypes.c_byte * 16)(
        0x10, 0xBF, 0xDC, 0xA5,   # Data1 LE
        0x30, 0x65,                 # Data2 LE
        0xD2, 0x11,                 # Data3 LE
        0x90, 0x1F,                 # Data4[0..1]
        0x00, 0xC0, 0x4F, 0xB9, 0x51, 0xED  # Data4[2..7]
    )
    guid = GUID_DEVINTERFACE_USB_DEVICE

    h_info = setupapi.SetupDiGetClassDevsW(
        ctypes.byref(guid), None, None,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE
    )
    # SetupDiGetClassDevs returns INVALID_HANDLE_VALUE (-1) on failure
    if h_info is None or h_info == -1:
        return None
    # On 64-bit, also check the unsigned form
    try:
        if h_info == ctypes.c_void_p(-1).value:
            return None
    except Exception:
        pass

    iface_data = SP_DEVICE_INTERFACE_DATA()
    iface_data.cbSize = ctypes.sizeof(SP_DEVICE_INTERFACE_DATA)
    target = f"vid_{vid:04x}&pid_{pid:04x}".lower()

    idx = 0
    while True:
        ok = setupapi.SetupDiEnumDeviceInterfaces(
            h_info, None, ctypes.byref(guid), idx, ctypes.byref(iface_data)
        )
        if not ok:
            break
        idx += 1

        # Get required buffer size
        buf_size = wt.DWORD(0)
        setupapi.SetupDiGetDeviceInterfaceDetailW(
            h_info, ctypes.byref(iface_data), None, 0,
            ctypes.byref(buf_size), None
        )
        if buf_size.value == 0:
            continue

        # Allocate buffer  (SP_DEVICE_INTERFACE_DETAIL_DATA: cbSize DWORD + WCHAR[])
        buf = ctypes.create_string_buffer(buf_size.value)
        # cbSize is platform-dependent: 8 on x64, 6 on x86
        cb = 8 if ctypes.sizeof(ctypes.c_void_p) == 8 else 6
        struct.pack_into("<I", buf, 0, cb)

        ok = setupapi.SetupDiGetDeviceInterfaceDetailW(
            h_info, ctypes.byref(iface_data), buf, buf_size,
            None, None
        )
        if not ok:
            continue

        # Extract wide-char device path (starts at offset 4, after cbSize)
        path_bytes = buf.raw[4:]
        path = path_bytes.decode("utf-16-le", errors="ignore").split("\x00")[0]

        if target in path.lower():
            setupapi.SetupDiDestroyDeviceInfoList(h_info)
            return path

    setupapi.SetupDiDestroyDeviceInfoList(h_info)
    return None


class WinUsbDevice:
    """Low-level WinUSB handle wrapper with thread-safe I/O."""

    def __init__(self, path: str):
        self._lock = threading.Lock()  # Protect all USB I/O from concurrent access

        # Open without FILE_FLAG_OVERLAPPED for synchronous I/O
        self._file_h = kernel32.CreateFileW(
            path,
            GENERIC_WRITE | GENERIC_READ,
            FILE_SHARE_WRITE | FILE_SHARE_READ,
            None, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None
        )
        # Check for INVALID_HANDLE_VALUE
        if self._file_h is None or self._file_h == -1:
            err = ctypes.GetLastError()
            raise OSError(f"CreateFileW failed for {path} (error {err})")
        try:
            if self._file_h == ctypes.c_void_p(-1).value:
                err = ctypes.GetLastError()
                raise OSError(f"CreateFileW failed for {path} (error {err})")
        except TypeError:
            pass

        self._usb_h = ctypes.c_void_p()
        ok = winusb.WinUsb_Initialize(self._file_h, ctypes.byref(self._usb_h))
        if not ok:
            err = ctypes.GetLastError()
            kernel32.CloseHandle(self._file_h)
            raise OSError(f"WinUsb_Initialize failed (error {err})")

        # Set a reasonable bulk read timeout (5 seconds) to avoid infinite hangs
        timeout_val = wt.ULONG(5000)
        winusb.WinUsb_SetPipePolicy(
            self._usb_h, PIPE_BULK_IN, PIPE_TRANSFER_TIMEOUT,
            ctypes.sizeof(timeout_val), ctypes.byref(timeout_val)
        )

    def control_transfer(self, request_type: int, request: int,
                         value: int, index: int,
                         data: bytes | bytearray | None = None,
                         length: int = 0) -> bytes:
        """Execute a USB control transfer (read or write), thread-safe."""
        transferred = wt.ULONG(0)

        with self._lock:
            if request_type & 0x80:  # Device-to-host (IN)
                if length <= 0:
                    return b""
                pkt = _pack_setup_packet(request_type, request, value, index, length)
                buf = ctypes.create_string_buffer(length)
                ok = winusb.WinUsb_ControlTransfer(
                    self._usb_h, pkt, buf, wt.ULONG(length),
                    ctypes.byref(transferred), None
                )
                if not ok:
                    err = ctypes.GetLastError()
                    raise OSError(f"WinUsb_ControlTransfer IN failed (error {err})")
                return buf.raw[:transferred.value]
            else:  # Host-to-device (OUT)
                if data is None or len(data) == 0:
                    pkt = _pack_setup_packet(request_type, request, value, index, 0)
                    ok = winusb.WinUsb_ControlTransfer(
                        self._usb_h, pkt, None, wt.ULONG(0),
                        ctypes.byref(transferred), None
                    )
                else:
                    pkt = _pack_setup_packet(request_type, request, value, index, len(data))
                    c_data = ctypes.create_string_buffer(bytes(data), len(data))
                    ok = winusb.WinUsb_ControlTransfer(
                        self._usb_h, pkt, c_data, wt.ULONG(len(data)),
                        ctypes.byref(transferred), None
                    )
                if not ok:
                    err = ctypes.GetLastError()
                    raise OSError(f"WinUsb_ControlTransfer OUT failed (error {err})")
                return b""

    def bulk_read(self, endpoint: int, length: int) -> bytes:
        """Read bulk data from USB endpoint, thread-safe."""
        buf = ctypes.create_string_buffer(length)
        transferred = wt.ULONG(0)
        with self._lock:
            ok = winusb.WinUsb_ReadPipe(
                self._usb_h, ctypes.c_ubyte(endpoint), buf, wt.ULONG(length),
                ctypes.byref(transferred), None
            )
        if not ok:
            err = ctypes.GetLastError()
            raise OSError(f"WinUsb_ReadPipe failed (error {err})")
        return buf.raw[:transferred.value]

    def close(self):
        try:
            winusb.WinUsb_Free(self._usb_h)
        except Exception:
            pass
        try:
            kernel32.CloseHandle(self._file_h)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Section 2 – RTL2832U Register Protocol
# ---------------------------------------------------------------------------

# USB control transfer directions (USB vendor request)
CTRL_IN  = 0xC0  # Vendor, device-to-host
CTRL_OUT = 0x40  # Vendor, host-to-device

# Register block IDs (from librtlsdr)
DEMODB = 0
USBB   = 1
SYSB   = 2
TUNB   = 3
ROMB   = 4
IRB    = 5
IICB   = 6

# USB block registers
USB_SYSCTL       = 0x2000
USB_CTRL         = 0x2010
USB_STAT         = 0x2014
USB_EPA_CFG      = 0x2144
USB_EPA_CTL      = 0x2148
USB_EPA_MAXPKT   = 0x2158
USB_EPA_MAXPKT_2 = 0x215a
USB_EPA_FIFO_CFG = 0x2160

# SYS block registers
DEMOD_CTL   = 0x3000
GPO         = 0x3001
GPI         = 0x3002
GPOE        = 0x3003
GPD         = 0x3004
SYSINTE     = 0x3005
SYSINTS     = 0x3006
GP_CFG0     = 0x3007
GP_CFG1     = 0x3008
SYSINTE_1   = 0x3009
SYSINTS_1   = 0x300a
DEMOD_CTL_1 = 0x300b
IR_SUSPEND  = 0x300c

# Default RTL crystal frequency
RTL_XTAL_FREQ = 28800000  # 28.8 MHz

# Default FIR coefficients (from librtlsdr, used for DAB/FM)
FIR_DEFAULT = [
    -54, -36, -41, -40, -32, -14,  14,  53,   # 8-bit signed
    101, 156, 215, 273, 327, 372, 404, 421     # 12-bit signed
]


class Rtl2832u:
    """Driver for the RTL2832U demodulator chip."""

    def __init__(self, usb: WinUsbDevice):
        self._usb = usb
        self.fir = list(FIR_DEFAULT)
        self.rate = 0
        self.rtl_xtal = RTL_XTAL_FREQ
        self.tun_xtal = RTL_XTAL_FREQ
        self.corr_ppm = 0

    # ---- Low-level register R/W ----

    def read_reg(self, block: int, addr: int, length: int) -> int:
        index = block << 8
        data = self._usb.control_transfer(CTRL_IN, 0, addr, index, length=length)
        if length == 1:
            return data[0]
        return (data[1] << 8) | data[0]

    def write_reg(self, block: int, addr: int, val: int, length: int):
        index = (block << 8) | 0x10
        if length == 1:
            data = bytes([val & 0xFF])
        else:
            data = bytes([(val >> 8) & 0xFF, val & 0xFF])
        self._usb.control_transfer(CTRL_OUT, 0, addr, index, data=data)

    def read_array(self, block: int, addr: int, length: int) -> bytes:
        index = block << 8
        return self._usb.control_transfer(CTRL_IN, 0, addr, index, length=length)

    def write_array(self, block: int, addr: int, data: bytes):
        index = (block << 8) | 0x10
        self._usb.control_transfer(CTRL_OUT, 0, addr, index, data=data)

    # ---- Demod register R/W (paged) ----

    def demod_read_reg(self, page: int, addr: int, length: int = 1) -> int:
        index = page
        usb_addr = (addr << 8) | 0x20
        data = self._usb.control_transfer(CTRL_IN, 0, usb_addr, index, length=length)
        if length == 1:
            return data[0]
        return (data[1] << 8) | data[0]

    def demod_write_reg(self, page: int, addr: int, val: int, length: int = 1):
        index = 0x10 | page
        usb_addr = (addr << 8) | 0x20
        if length == 1:
            data = bytes([val & 0xFF])
        else:
            data = bytes([(val >> 8) & 0xFF, val & 0xFF])
        self._usb.control_transfer(CTRL_OUT, 0, usb_addr, index, data=data)
        # Dummy read to flush write (matches librtlsdr behavior)
        self.demod_read_reg(0x0A, 0x01, 1)

    # ---- I2C bridge (for talking to the R820T2 tuner) ----

    def set_i2c_repeater(self, on: bool):
        """Open/close the I2C repeater gate to the tuner chip."""
        # This uses demod page 1, reg 0x01. We bypass demod_write_reg to
        # avoid the recursive dummy-read flush (which itself calls demod_read_reg).
        index = 0x10 | 1  # page 1, write
        usb_addr = (0x01 << 8) | 0x20
        val = 0x18 if on else 0x10
        self._usb.control_transfer(CTRL_OUT, 0, usb_addr, index, data=bytes([val]))

    def i2c_write(self, i2c_addr: int, data: bytes):
        self.write_array(IICB, i2c_addr, data)

    def i2c_read(self, i2c_addr: int, length: int) -> bytes:
        return self.read_array(IICB, i2c_addr, length)

    # ---- FIR filter setup ----

    def set_fir(self):
        fir_bytes = bytearray(20)
        # First 8 coefficients as int8
        for i in range(8):
            fir_bytes[i] = self.fir[i] & 0xFF
        # Next 8 coefficients packed as int12 (pairs)
        for i in range(0, 8, 2):
            v0 = self.fir[8 + i]
            v1 = self.fir[8 + i + 1]
            off = 8 + (i * 3) // 2
            fir_bytes[off]     = (v0 >> 4) & 0xFF
            fir_bytes[off + 1] = ((v0 << 4) | ((v1 >> 8) & 0x0F)) & 0xFF
            fir_bytes[off + 2] = v1 & 0xFF
        for i in range(20):
            self.demod_write_reg(1, 0x1C + i, fir_bytes[i], 1)

    # ---- Baseband initialization (from rtlsdr_init_baseband) ----

    def init_baseband(self):
        # USB block init
        self.write_reg(USBB, USB_SYSCTL, 0x09, 1)
        self.write_reg(USBB, USB_EPA_MAXPKT, 0x0002, 2)
        self.write_reg(USBB, USB_EPA_CTL, 0x1002, 2)

        # Power on demod
        self.write_reg(SYSB, DEMOD_CTL_1, 0x22, 1)
        self.write_reg(SYSB, DEMOD_CTL, 0xE8, 1)

        # Reset demod (toggle soft_rst bit)
        self.demod_write_reg(1, 0x01, 0x14, 1)
        self.demod_write_reg(1, 0x01, 0x10, 1)

        # Disable spectrum inversion and adjacent channel rejection
        self.demod_write_reg(1, 0x15, 0x00, 1)
        self.demod_write_reg(1, 0x16, 0x0000, 2)

        # Clear DDC shift and IF freq registers
        for i in range(6):
            self.demod_write_reg(1, 0x16 + i, 0x00, 1)

        # Load FIR filter
        self.set_fir()

        # Enable SDR mode, disable DAGC
        self.demod_write_reg(0, 0x19, 0x05, 1)

        # Init FSM state-holding register
        self.demod_write_reg(1, 0x93, 0xF0, 1)
        self.demod_write_reg(1, 0x94, 0x0F, 1)

        # Disable AGC
        self.demod_write_reg(1, 0x11, 0x00, 1)
        self.demod_write_reg(1, 0x04, 0x00, 1)

        # Disable PID filter
        self.demod_write_reg(0, 0x61, 0x60, 1)

        # Default ADC datapath
        self.demod_write_reg(0, 0x06, 0x80, 1)

        # Enable Zero-IF, DC cancellation, IQ estimation
        self.demod_write_reg(1, 0xB1, 0x1B, 1)

        # Disable 4.096 MHz clock output
        self.demod_write_reg(0, 0x0D, 0x83, 1)

    # ---- Sample rate ----

    def set_sample_rate(self, rate: int):
        if rate <= 0:
            raise ValueError(f"Invalid sample rate: {rate}")
        xtal = self._corrected_xtal()

        # rsamp_ratio = floor(xtal * 2^22 / rate), aligned to 4
        rsamp_ratio = (xtal * (1 << 22)) // rate
        rsamp_ratio &= 0x0FFFFFFC

        if rsamp_ratio == 0:
            raise ValueError(f"Sample rate {rate} too high for crystal {xtal}")

        real_rate = (xtal * (1 << 22)) // rsamp_ratio
        print(f"[rtl2832u] Sample rate: {real_rate} Hz (requested {rate})")

        # Write 28-bit rsamp_ratio as two 16-bit demod registers
        self.demod_write_reg(1, 0x9F, (rsamp_ratio >> 16) & 0xFFFF, 2)
        self.demod_write_reg(1, 0xA1, rsamp_ratio & 0xFFFF, 2)

        # Reset demod
        self.demod_write_reg(1, 0x01, 0x14, 1)
        self.demod_write_reg(1, 0x01, 0x10, 1)

        # Set IF frequency to 0 (Zero-IF mode)
        self._set_if_freq(0)

        self.rate = real_rate
        return real_rate

    def _set_if_freq(self, freq: int):
        xtal = self._corrected_xtal()
        if freq == 0:
            if_freq = 0
        else:
            if_freq = int(((freq * (1 << 22)) / xtal) * -1)
        # Mask to 22-bit two's complement
        if_freq &= 0x3FFFFF
        self.demod_write_reg(1, 0x19, (if_freq >> 16) & 0x3F, 1)
        self.demod_write_reg(1, 0x1A, (if_freq >> 8) & 0xFF, 1)
        self.demod_write_reg(1, 0x1B, if_freq & 0xFF, 1)

    def _corrected_xtal(self) -> int:
        return int(self.rtl_xtal * (1.0 + self.corr_ppm / 1e6))

    # ---- Streaming control ----

    def reset_endpoint(self):
        self.write_reg(USBB, USB_EPA_CTL, 0x1002, 2)

    def reset_demod(self):
        self.demod_write_reg(1, 0x01, 0x14, 1)
        self.demod_write_reg(1, 0x01, 0x10, 1)

    def read_iq(self, length: int = 512 * 32) -> bytes:
        """Read a chunk of raw I/Q bytes from USB bulk endpoint."""
        return self._usb.bulk_read(PIPE_BULK_IN, length)


# ---------------------------------------------------------------------------
# Section 3 – R820T/R820T2 Tuner Driver
# ---------------------------------------------------------------------------

R820T_I2C_ADDR   = 0x34   # 7-bit 0x1A, left-shifted by 1
R828D_I2C_ADDR   = 0x74
REG_SHADOW_START = 0x05
NUM_REGS         = 0x1F - REG_SHADOW_START + 1  # 27 registers (0x05..0x1F)

R82XX_INIT_REGS = bytes([
    0x83, 0x32, 0x75,               # 05..07
    0xC0, 0x40, 0xD6, 0x6C,         # 08..0B
    0xF5, 0x63, 0x75, 0x68,         # 0C..0F
    0x6C, 0x83, 0x80, 0x00,         # 10..13
    0x0F, 0x00, 0xC0, 0x30,         # 14..17
    0x48, 0xCC, 0x60, 0x00,         # 18..1B
    0x54, 0xAE, 0x4A, 0xC0,         # 1C..1F
])

# Frequency range table: (freq_MHz, open_d, rf_mux_ploy, tf_c, cap20p, cap10p, cap0p)
FREQ_RANGES = [
    (0,   0x08, 0x02, 0xDF, 0x02, 0x01, 0x00),
    (50,  0x08, 0x02, 0xBE, 0x02, 0x01, 0x00),
    (55,  0x08, 0x02, 0x8B, 0x02, 0x01, 0x00),
    (60,  0x08, 0x02, 0x7B, 0x02, 0x01, 0x00),
    (65,  0x08, 0x02, 0x69, 0x02, 0x01, 0x00),
    (70,  0x08, 0x02, 0x58, 0x02, 0x01, 0x00),
    (75,  0x00, 0x02, 0x44, 0x02, 0x01, 0x00),
    (80,  0x00, 0x02, 0x44, 0x02, 0x01, 0x00),
    (90,  0x00, 0x02, 0x34, 0x01, 0x01, 0x00),
    (100, 0x00, 0x02, 0x34, 0x01, 0x01, 0x00),
    (110, 0x00, 0x02, 0x24, 0x01, 0x01, 0x00),
    (120, 0x00, 0x02, 0x24, 0x01, 0x01, 0x00),
    (140, 0x00, 0x02, 0x14, 0x01, 0x01, 0x00),
    (180, 0x00, 0x02, 0x13, 0x00, 0x00, 0x00),
    (220, 0x00, 0x02, 0x13, 0x00, 0x00, 0x00),
    (250, 0x00, 0x02, 0x11, 0x00, 0x00, 0x00),
    (280, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00),
    (310, 0x00, 0x41, 0x00, 0x00, 0x00, 0x00),
    (450, 0x00, 0x41, 0x00, 0x00, 0x00, 0x00),
    (588, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00),
    (650, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00),
]

# LNA gain steps (index -> tenth-dB) from librtlsdr
R82XX_LNA_GAIN_STEPS   = [0, 9, 13, 40, 38, 13, 31, 22, 26, 31, 26, 14, 19, 5, 35, 13]
R82XX_MIXER_GAIN_STEPS = [0, 5, 10, 10, 19, 9, 10, 25, 17, 10, 8, 16, 13, 6, 3, -8, -6]

# Precomputed bit-reversal LUT (R820T2 reads come back bit-reversed)
_BITREV_LUT = bytes(
    ((((b >> 0) & 1) << 7) | (((b >> 1) & 1) << 6) | (((b >> 2) & 1) << 5) |
     (((b >> 3) & 1) << 4) | (((b >> 4) & 1) << 3) | (((b >> 5) & 1) << 2) |
     (((b >> 6) & 1) << 1) | (((b >> 7) & 1) << 0))
    for b in range(256)
)


class R820T2:
    """Driver for the Rafael Micro R820T / R820T2 silicon tuner."""

    def __init__(self, rtl: Rtl2832u, i2c_addr: int = R820T_I2C_ADDR,
                 xtal: int = RTL_XTAL_FREQ):
        self._rtl = rtl
        self._addr = i2c_addr
        self._xtal = xtal
        # Shadow registers (0x05..0x1F)
        self._regs = bytearray(R82XX_INIT_REGS)

    # ---- I2C R/W (via RTL2832U I2C bridge) ----

    # RTL2832U I2C master supports max 8-byte transfers.
    # First byte is the register address, leaving 7 bytes for data payload.
    _MAX_I2C_MSG_LEN = 8
    _MAX_I2C_DATA    = _MAX_I2C_MSG_LEN - 1  # 7 payload bytes per transfer

    def _write(self, reg: int, data: bytes | bytearray):
        """Write to tuner registers via I2C bridge (auto-chunked)."""
        # Update shadow copy
        r = reg - REG_SHADOW_START
        for i, b in enumerate(data):
            if 0 <= r + i < NUM_REGS:
                self._regs[r + i] = b

        # The RTL2832U I2C master limits transfers to 8 bytes total
        # (1 byte register address + up to 7 bytes data).  Chunk if needed.
        pos = 0
        cur_reg = reg
        while pos < len(data):
            chunk_len = min(self._MAX_I2C_DATA, len(data) - pos)
            msg = bytes([cur_reg]) + bytes(data[pos:pos + chunk_len])
            self._rtl.i2c_write(self._addr, msg)
            pos += chunk_len
            cur_reg += chunk_len

    def _write_reg(self, reg: int, val: int):
        self._write(reg, bytes([val & 0xFF]))

    def _write_reg_mask(self, reg: int, val: int, mask: int):
        r = reg - REG_SHADOW_START
        if 0 <= r < NUM_REGS:
            old = self._regs[r]
        else:
            old = 0
        new_val = (old & ~mask) | (val & mask)
        self._write(reg, bytes([new_val & 0xFF]))

    def _read(self, reg: int, length: int) -> bytes:
        """Read from tuner registers (data comes bit-reversed from the chip)."""
        self._rtl.i2c_write(self._addr, bytes([reg]))
        raw = self._rtl.i2c_read(self._addr, length)
        return bytes(_BITREV_LUT[b] for b in raw)

    # ---- Initialization ----

    def init(self):
        """Initialize R820T2 with default register values."""
        self._rtl.set_i2c_repeater(True)
        try:
            # Write all shadow registers (0x05..0x1F)
            self._regs = bytearray(R82XX_INIT_REGS)
            self._write(REG_SHADOW_START, R82XX_INIT_REGS)
        finally:
            self._rtl.set_i2c_repeater(False)
        print("[r820t2] Tuner initialized")

    # ---- Set frequency ----

    def set_freq(self, freq_hz: int):
        """Tune to the given frequency in Hz."""
        self._rtl.set_i2c_repeater(True)
        try:
            self._set_mux(freq_hz)
            self._set_pll(freq_hz)
        finally:
            self._rtl.set_i2c_repeater(False)
        # Flush any stale data from the previous frequency
        self._rtl.reset_endpoint()
        print(f"[r820t2] Tuned to {freq_hz / 1e6:.3f} MHz")

    def _set_mux(self, freq_hz: int):
        """Select the right RF mux / tracking filter for the frequency band."""
        freq_mhz = freq_hz // 1_000_000
        # Find the matching frequency range entry
        rng = FREQ_RANGES[0]
        for i in range(len(FREQ_RANGES) - 1):
            if freq_mhz < FREQ_RANGES[i + 1][0]:
                rng = FREQ_RANGES[i]
                break
        else:
            rng = FREQ_RANGES[-1]

        _, open_d, rf_mux, tf_c, cap20, cap10, cap0 = rng

        # Open drain
        self._write_reg_mask(0x17, open_d, 0x08)
        # RF_MUX,Polymux
        self._write_reg_mask(0x1A, rf_mux, 0xC3)
        # TF BAND
        self._write_reg(0x1B, tf_c)
        # XTAL cap (default: low cap 0pF + drive)
        val = cap0 | 0x08
        self._write_reg_mask(0x10, val, 0x0B)
        # Clear RF/IF gain power detector
        self._write_reg_mask(0x08, 0x00, 0x3F)
        self._write_reg_mask(0x09, 0x00, 0x3F)

    def _set_pll(self, freq_hz: int):
        """Program the R820T2 fractional-N PLL to the desired frequency."""
        freq_khz = (freq_hz + 500) // 1000
        pll_ref = self._xtal  # Hz

        # PLL autotune = 128 kHz
        self._write_reg_mask(0x1A, 0x00, 0x0C)

        # --- Find the correct mix_div (VCO divider) ---
        # VCO range: 1770 MHz .. 3540 MHz (expressed in kHz for comparison)
        vco_min = 1770000  # kHz
        vco_max = vco_min * 2  # 3540000 kHz
        mix_div = 2
        div_num = 0

        while mix_div <= 64:
            prod = freq_khz * mix_div
            if vco_min <= prod < vco_max:
                # Count log2(mix_div) - 1
                div_buf = mix_div
                while div_buf > 2:
                    div_buf >>= 1
                    div_num += 1
                break
            mix_div <<= 1
        else:
            print(f"[r820t2] WARNING: No valid mix_div for {freq_hz} Hz!")
            return

        # Read VCO fine-tune indicator from chip status register
        data = self._read(0x00, 5)
        vco_fine_tune = (data[4] & 0x30) >> 4
        vco_power_ref = 2  # R820T reference value (R828D uses 1)
        if vco_fine_tune > vco_power_ref:
            div_num = max(0, div_num - 1)
        elif vco_fine_tune < vco_power_ref:
            div_num = min(5, div_num + 1)  # max div_num for 6 bits = 5

        # Write div_num to reg 0x10 bits [7:5]
        self._write_reg_mask(0x10, (div_num << 5) & 0xE0, 0xE0)

        # --- Calculate PLL integer (nint) and fractional (vco_fra) parts ---
        vco_freq = freq_hz * mix_div  # Hz (Python handles big ints)
        nint = vco_freq // (2 * pll_ref)
        vco_fra = vco_freq - 2 * pll_ref * nint

        # If nint > 63, switch to reference divider /2 mode
        if nint > 63:
            nint = vco_freq // pll_ref
            vco_fra = vco_freq - pll_ref * nint
            self._write_reg_mask(0x10, 0x10, 0x10)  # refdiv2 = 1
            pll_ref_sdm = pll_ref
        else:
            self._write_reg_mask(0x10, 0x00, 0x10)  # refdiv2 = 0
            pll_ref_sdm = 2 * pll_ref

        # Clamp nint to valid range (13..76 per R820T2 datasheet)
        nint = max(13, min(76, nint))

        # Decompose nint into NI and SI fields
        # nint = 4*ni + si + 13, where si is in [0..3]
        ni = (nint - 13) // 4
        si = (nint - 13) % 4   # Use modulo instead of subtraction to avoid bugs
        self._write(0x14, bytes([ni + (si << 6)]))

        # VCO current = 100
        self._write_reg_mask(0x12, 0x80, 0xE0)

        # --- SDM (Sigma-Delta Modulator) fractional part ---
        if vco_fra == 0:
            # Integer-N mode
            self._write_reg_mask(0x12, 0x08, 0x08)
            sdm = 0
        else:
            # Fractional-N mode
            self._write_reg_mask(0x12, 0x00, 0x08)
            # 16-bit SDM value
            sdm = (65536 * vco_fra) // (2 * pll_ref_sdm)
            sdm = max(0, min(65535, sdm))

        self._write(0x16, bytes([(sdm >> 8) & 0xFF, sdm & 0xFF]))

        # --- Wait for PLL lock ---
        time.sleep(0.010)  # 10 ms
        data = self._read(0x00, 3)
        if not (data[2] & 0x40):
            print("[r820t2] WARNING: PLL not locked, increasing VCO current ...")
            self._write_reg_mask(0x12, 0x60, 0xE0)
            time.sleep(0.010)
            data = self._read(0x00, 3)
            if not (data[2] & 0x40):
                print("[r820t2] ERROR: PLL still not locked!")

    # ---- Gain control ----

    def set_gain(self, gain_tenth_db: int):
        """Set tuner gain in tenths of dB. 0 = automatic gain control."""
        self._rtl.set_i2c_repeater(True)
        try:
            if gain_tenth_db == 0:
                # Auto gain
                self._write_reg_mask(0x0C, 0x00, 0x08)  # LNA AGC on
                self._write_reg_mask(0x07, 0x10, 0x10)   # Mixer AGC on
                return

            # Manual gain mode
            self._write_reg_mask(0x0C, 0x08, 0x08)  # LNA AGC off (manual)
            self._write_reg_mask(0x07, 0x00, 0x10)   # Mixer AGC off (manual)

            # Walk the LNA gain table to find the best index
            total = 0
            lna_idx = 0
            for i in range(1, 16):
                total += R82XX_LNA_GAIN_STEPS[i]
                if total >= gain_tenth_db:
                    lna_idx = i
                    break
            else:
                lna_idx = 15

            remaining = max(0, gain_tenth_db - sum(R82XX_LNA_GAIN_STEPS[1:lna_idx + 1]))

            # Walk the Mixer gain table for the remainder
            total = 0
            mix_idx = 0
            for i in range(1, len(R82XX_MIXER_GAIN_STEPS)):
                total += R82XX_MIXER_GAIN_STEPS[i]
                if total >= remaining:
                    mix_idx = i
                    break
            else:
                mix_idx = len(R82XX_MIXER_GAIN_STEPS) - 1

            self._write_reg_mask(0x05, lna_idx, 0x0F)
            self._write_reg_mask(0x07, mix_idx, 0x0F)
        finally:
            self._rtl.set_i2c_repeater(False)

    def set_bandwidth(self, bw_hz: int):
        """Set the IF filter bandwidth (simplified: narrow < 300 kHz, wide otherwise)."""
        self._rtl.set_i2c_repeater(True)
        try:
            if bw_hz < 300000:
                # Narrow bandwidth (AM/SSB)
                self._write_reg_mask(0x0A, 0x0F, 0x0F)
                self._write_reg_mask(0x0B, 0x6B, 0xEF)
            else:
                # Wide bandwidth (FM)
                self._write_reg_mask(0x0A, 0x0F, 0x0F)
                self._write_reg_mask(0x0B, 0x60, 0xEF)
        finally:
            self._rtl.set_i2c_repeater(False)


# ---------------------------------------------------------------------------
# Section 4 – TCP Server (rtl_tcp compatible protocol)
# ---------------------------------------------------------------------------

RTL_TCP_MAGIC = b"RTL0"
TUNER_R820T = 5  # Tuner type enum from rtl_tcp

# rtl_tcp command IDs
CMD_SET_FREQ        = 0x01
CMD_SET_SAMPLERATE  = 0x02
CMD_SET_GAIN_MODE   = 0x03
CMD_SET_GAIN        = 0x04
CMD_SET_FREQ_CORR   = 0x05
CMD_SET_IF_GAIN     = 0x06
CMD_SET_AGC_MODE    = 0x08
CMD_SET_DIRECT_SAMP = 0x09
CMD_SET_OFFSET_TUNE = 0x0A
CMD_SET_RTL_XTAL    = 0x0B
CMD_SET_TUNER_XTAL  = 0x0C
CMD_SET_TUNER_BW    = 0x0E


def _build_header(tuner_type: int = TUNER_R820T, gain_count: int = 29) -> bytes:
    """Build the 12-byte rtl_tcp handshake header."""
    return RTL_TCP_MAGIC + struct.pack(">II", tuner_type, gain_count)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a socket (TCP may fragment)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return bytes(buf)  # Connection closed
        buf.extend(chunk)
    return bytes(buf)


def serve_tcp(rtl: Rtl2832u, tuner: R820T2, port: int):
    """Run the rtl_tcp-compatible streaming server."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"[tcp] Listening on 0.0.0.0:{port} ...")

    while True:
        conn, addr = srv.accept()
        print(f"[tcp] Client connected: {addr}")
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.sendall(_build_header())
        except OSError:
            conn.close()
            continue

        # Start I/Q streaming in a background thread
        stop_event = threading.Event()
        # Capture conn in default arg to avoid closure issues with rebinding
        client_conn = conn

        def _stream(c=client_conn):
            try:
                while not stop_event.is_set():
                    try:
                        data = rtl.read_iq(16384)
                        if data:
                            c.sendall(data)
                    except OSError:
                        break
            except Exception as e:
                if not stop_event.is_set():
                    print(f"[tcp] Stream error: {e}")

        t = threading.Thread(target=_stream, daemon=True)
        t.start()

        # Command reception loop
        try:
            while True:
                cmd_buf = _recv_exact(conn, 5)
                if len(cmd_buf) < 5:
                    break  # Client disconnected
                cmd_id = cmd_buf[0]
                value  = struct.unpack(">I", cmd_buf[1:5])[0]

                try:
                    if cmd_id == CMD_SET_FREQ:
                        tuner.set_freq(value)
                    elif cmd_id == CMD_SET_SAMPLERATE:
                        rtl.set_sample_rate(value)
                        rtl.reset_demod()
                    elif cmd_id == CMD_SET_GAIN:
                        tuner.set_gain(value)
                    elif cmd_id == CMD_SET_GAIN_MODE:
                        if value == 0:
                            tuner.set_gain(0)  # auto
                    elif cmd_id == CMD_SET_FREQ_CORR:
                        # Value is signed PPM (passed as unsigned 32-bit)
                        if value > 0x7FFFFFFF:
                            value -= 0x100000000
                        rtl.corr_ppm = value
                    elif cmd_id == CMD_SET_TUNER_BW:
                        tuner.set_bandwidth(value)
                    elif cmd_id == CMD_SET_AGC_MODE:
                        if value == 1:
                            tuner.set_gain(0)   # enable AGC
                except Exception as e:
                    print(f"[tcp] Command {cmd_id:#x} error: {e}")
        except (ConnectionError, OSError):
            pass

        stop_event.set()
        t.join(timeout=2)
        try:
            conn.close()
        except Exception:
            pass
        print(f"[tcp] Client disconnected: {addr}")


# ---------------------------------------------------------------------------
# Section 5 – Main entry point
# ---------------------------------------------------------------------------

# Known RTL-SDR VID/PID pairs (from librtlsdr)
KNOWN_DEVICES = [
    (0x0BDA, 0x2832, "Generic RTL2832U"),
    (0x0BDA, 0x2838, "Generic RTL2832U OEM"),
]


def find_dongle() -> tuple[str, str]:
    """Find the first connected RTL-SDR dongle, return (device_path, name)."""
    for vid, pid, name in KNOWN_DEVICES:
        path = _find_device_path(vid, pid)
        if path:
            return path, name
    raise RuntimeError(
        "No RTL-SDR device found!\n"
        "Make sure:\n"
        "  1. The dongle is plugged in.\n"
        "  2. You installed the WinUSB driver via Zadig.\n"
        "     (https://zadig.akeo.ie/)"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Pure-Python RTL-SDR USB driver (replaces rtl_tcp.exe)"
    )
    ap.add_argument("--freq", type=int, default=100000000,
                    help="Initial center frequency in Hz (default: 100 MHz)")
    ap.add_argument("--rate", type=int, default=2048000,
                    help="Sample rate in Hz (default: 2.048 MSPS)")
    ap.add_argument("--gain", type=int, default=0,
                    help="Tuner gain in tenths of dB (0=auto, default: 0)")
    ap.add_argument("--port", type=int, default=1234,
                    help="TCP server port (default: 1234)")
    ap.add_argument("--test", action="store_true",
                    help="Test: enumerate USB and print dongle info, then exit")
    args = ap.parse_args()

    # Find dongle
    print("[main] Searching for RTL-SDR dongle ...")
    path, name = find_dongle()
    print(f"[main] Found: {name}")
    print(f"[main] Path:  {path}")

    if args.test:
        print("[main] Test mode — device found. Exiting.")
        return

    # Open USB device
    print("[main] Opening WinUSB device ...")
    usb = WinUsbDevice(path)

    # Initialize RTL2832U
    rtl = Rtl2832u(usb)
    print("[main] Initializing baseband ...")
    rtl.init_baseband()

    # Initialize R820T2 tuner
    tuner = R820T2(rtl)
    print("[main] Initializing tuner ...")
    tuner.init()

    # Configure frequency, sample rate, gain
    print(f"[main] Setting frequency: {args.freq / 1e6:.3f} MHz")
    tuner.set_freq(args.freq)

    print(f"[main] Setting sample rate: {args.rate} SPS")
    rtl.set_sample_rate(args.rate)

    if args.gain:
        print(f"[main] Setting gain: {args.gain / 10:.1f} dB")
        tuner.set_gain(args.gain)

    # Reset endpoint before streaming
    rtl.reset_endpoint()

    # Start TCP server
    try:
        serve_tcp(rtl, tuner, args.port)
    except KeyboardInterrupt:
        print("\n[main] Shutting down ...")
    finally:
        usb.close()
        print("[main] Done.")


if __name__ == "__main__":
    main()
