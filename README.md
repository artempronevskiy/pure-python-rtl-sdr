# 📡 Software Defined Radio (SDR) & Digital Signal Processing (DSP)
**A Comprehensive Masters-Level Course for Software Engineers**

*Welcome to the intersection of Software Engineering, Radio Frequency (RF) Physics, and Applied Mathematics.*

As a professional software engineer, you possess a deep intuitive understanding of discrete data, algorithms, and computational complexity. This course bridges the gap between software architecture and the physical electromagnetic spectrum. It provides a mathematically rigorous, ground-up explanation of Digital Signal Processing (DSP) and Software Defined Radio (SDR), proving the theorems through highly optimized, pure Python implementations.

---

## 📑 Course Syllabus

**Part I: The Physics & Foundations of Software Radio**
1.  **Hardware Architecture:** The Heterodyne Receiver & ADC Boundary
2.  **Sampling Theory:** The Nyquist-Shannon Theorem & Quadrature (I/Q) Sampling
3.  **Complex Mathematics:** Euler’s Formula, Analytic Signals, and Baseband Spectra

**Part II: The Core DSP Pipeline**
4.  **Frequency Translation:** Tuning via Complex Exponentials (Digital Down Conversion)
5.  **Digital Decimation & Filtering Theory:** FIR vs. IIR, Sinc Functions, and Anti-Aliasing
6.  **Control Systems:** Automatic Gain Control (AGC) Loop Dynamics

**Part III: Modulation & Demodulation Calculus**
7.  **Amplitude Modulation (AM):** Envelope Geometry & Diode Detectors in Code
8.  **Single Sideband (SSB):** The Hilbert Transform, Weaver Architecture, and Spectral Folding
9.  **Frequency Modulation (FM):** Non-Linear Angle Calculus & Polar Discrimination

---

# Part I: The Physics & Foundations of Software Radio

## 1. Hardware Architecture: The Analog-to-Digital Boundary

To manipulate radio waves in software, we must first safely digitize the electromagnetic field. Historically, a radio was an uninterrupted analog pipeline of voltage—from the antenna to the speaker diaphragm. A **Software Defined Radio (SDR)** moves the Analog-to-Digital Converter (ADC) as close to the antenna as physically and economically possible.

### The RTL-SDR Architecture (The R820T2 & RTL2832U)
A common $30 RTL-SDR dongle acts as our hardware physical layer. It consists of two primary ICs:
1.  **The Tuner (R820T2):** A low-noise analog chip. It amplifies the microscopic nanowatt signals from the antenna (LNA), then uses a **Mixer** and a Variable Frequency Oscillator (VFO) to shift a chunk of the radio spectrum (e.g., 100 MHz) down to a lower, fixed Intermediate Frequency (IF).
2.  **The ADC & Digital Down Converter (RTL2832U):** This chip samples the analog IF voltage at 28.8 Mega-Samples Per Second (MSPS). It then digitally shifts the signal a final time to exactly $0$ Hz (Baseband) and decimates the stream down to a manageable rate (e.g., 2.4 MSPS) to send to your computer via USB.

*The hardware's only job is to capture a chunk of the spectrum and digitize it. Everything else—filtering, tuning, demodulation—is now a software problem.*

---

## 2. Sampling Theory: Nyquist & Quadrature (I/Q)

### The Nyquist-Shannon Limit
When digitizing a continuous physical signal $x(t)$, how fast must the ADC run?

> **The Nyquist-Shannon Theorem:** To perfectly reconstruct a continuous-time signal containing frequencies up to $f_{max}$, the sampling rate $f_s$ must be strictly greater than $2 \cdot f_{max}$ ($f_s > 2f_{max}$).

If you want to observe a 1 MHz wide chunk of radio spectrum, traditional sampling requires an ADC running at $\ge 2$ MSPS. However, SDRs achieve a $f_s = \text{Bandwidth}$ ratio ($1\text{ MHz bandwidth} \rightarrow 1\text{ MSPS}$ rate) by utilizing **Quadrature Sampling**.

### The Problem with 1D Real Sampling
If you sample a real voltage $v(t) = A\cos(2\pi f t)$, you lose a critical dimension: **Phase**.
At any frozen sample point, you do not know if the physical wave is rising or falling. Mathematically, a pure real cosine contains both positive and negative frequencies simultaneously:
$$ \cos(\omega t) = \frac{e^{j\omega t} + e^{-j\omega t}}{2} $$
Because of this intrinsic symmetry, a 1D real sample stream cannot distinguish whether a signal is $+100$ kHz *above* the center frequency or $-100$ kHz *below* it. They overlap destructively.

### The Solution: Quadrature (I/Q) Sampling (2D)
We solve this by taking *two* discrete samples simultaneously. The incoming RF signal is duplicated and multiplied by two local oscillators offset perfectly by 90° ($\pi/2$ radians):
*   **I (In-phase):** Multiplied by $\cos(\omega_0 t)$
*   **Q (Quadrature):** Multiplied by $-\sin(\omega_0 t)$

We treat these two discrete volt streams as a single, unified sequence of **Complex Numbers**:
$$ z[n] = I[n] + jQ[n] $$

---

## 3. Complex Mathematics: The Analytic Signal

Why complex numbers? By Euler's formula: 
$$ e^{j\theta} = \cos(\theta) + j\sin(\theta) $$

An **Analytic Signal** is mathematically a single-sided rotating vector: $A e^{j(\omega t + \phi)}$. 
*   **Frequency ($\omega$):** If the vector rotates counter-clockwise, the frequency is positive. If it rotates clockwise, the frequency is negative.
*   **Amplitude ($A$):** The Euclidean length of the vector.
*   **Phase ($\phi$):** The instantaneous angle of the vector.

**Core Concept:** I/Q data is a 2D digital representation of a rotating electromagnetic vector. Because it has two axes, it perfectly separates positive and negative frequencies relative to your hardware tuning center ($0$ Hz).

### Python Implementation: Ingesting the UDP/TCP Stream
The RTL-SDR streams bare unsigned 8-bit interleaved integers: `[I0, Q0, I1, Q1, ...]`. Zero volts is the hardware midpoint (`127.5`).

```python
# Extracting complex baseband vectors from raw 8-bit USB bytes
n_pairs = len(raw_bytes) // 2
for k in range(n_pairs):
    
    # 1. Cast 8-bit unsigned to float
    # 2. Subtract 127.5 to re-center DC voltage at 0.0
    i_raw = raw_bytes[k * 2]     - 127.5  
    q_raw = raw_bytes[k * 2 + 1] - 127.5
    
    # Mathematically, we now possess the complex sequence:
    # z[k] = i_raw + j * q_raw
```

---
# Part II: The Core DSP Pipeline

Every Software Defined Radio application implements the exact same fundamental sequence:
1.  **Tune (Frequency Translate)**
2.  **Filter & Decimate (Downsample)**
3.  **Demodulate**
4.  **Audio Output & AGC**

## 4. Frequency Translation (Tuning in Software)

If your physical hardware is tuned to 100.0 MHz, but the radio station of interest is at 100.3 MHz, the signal sits at a digital baseband offset of $+300$ kHz. 

To filter and demodulate it accurately, we must center it at exactly $0$ Hz DC. We achieve this by multiplying the entire complex time-domain sequence by a complex exponential spinning at the inverse offset ($-300$ kHz). 

By exponent rules, multiplying in the time domain causes a strict linear shift in the frequency domain:
$$ z_{shifted}[n] = z[n] \cdot e^{-j 2\pi f_{shift} t} $$

Applying Euler's algebraic expansion:
$$ (I \cdot \cos\theta - Q \cdot \sin\theta) + j(I \cdot \sin\theta + Q \cdot \cos\theta) $$

```python
# Tuning an offset frequency back to 0 Hz Baseband
phase_step = 2.0 * math.pi * offset_hz / sample_rate
phase_accum += phase_step  

cos_p = math.cos(phase_accum)
sin_p = math.sin(phase_accum)

# Complex multiply by e^{j * phase_accum}
shifted_i = i_raw * cos_p - q_raw * sin_p
shifted_q = i_raw * sin_p + q_raw * cos_p
```

---

## 5. Digital Decimation & Anti-Aliasing Theory

SDRs generate massive wideband data arrays. For AM radio, we may sample at 240,000 Samples Per Second (SPS). A computer soundcard legally requires audio rates (e.g., 48,000 SPS). We must mathematically downsample **(Decimate)** the array by a factor of $D = 5$.

### The Aliasing Crisis
If you simply `array[::5]` (take every 5th sample), any physical RF noise or adjacent stations present at frequencies $> 24$ kHz ($f_{new\_rate}/2$) will instantaneously physically "fold" back into your audio band. This is **Aliasing**, and it sounds like brutal, irrecoverable hiss.

**The Golden Rule of Decimation:** Before decimating by factor $D$, you *must* apply a Low-Pass Filter (LPF) rejecting all frequencies above the new Nyquist limit.

### FIR vs. IIR Filters in Software
1.  **FIR (Finite Impulse Response):** Mathematically a sliding dot-product convolution ($y[n] = \sum h[k]x[n-k]$). Intrinsically stable, strictly linear phase (it delays all audio frequencies equally, preserving waveform shape). Computationally expensive ($O(N)$).
2.  **IIR (Infinite Impulse Response):** Similar to analog RC circuits. Utilizes recursive feedback ($y[n] = x[n] + \alpha y[n-1]$). Extremely cheap ($O(1)$), but non-linear phase (warps audio transients).

### The Box-car Integrator (Moving Average)
To maximize pure-Python efficiency, we combine anti-alias FIR filtering and decimation into a single step called a **Box-car Integrator**.

Averaging $N$ contiguous samples is mathematically identical to convolving the signal with a rectangular pulse in the time domain. By the Fourier Transform, a rectangle in time is a `sinc` ($\frac{\sin(x)}{x}$) function in the frequency domain. A `sinc` filter provides deep attenuation nulls *exactly* at the integer multiples of the decimation rate—precisely where aliasing occurs!

```python
# Decimate by factor of 5 (240 kSPS -> 48 kSPS)
# Provides simultaneous anti-aliasing (FIR sinc filter) and downsampling

n_decimated = len(i_full) - (len(i_full) % 5)
inv5 = 1.0 / 5.0

audio_baseband = [
    # Summing 5 contiguous wideband samples, multiplying by 1/5
    (i_full[j] + i_full[j+1] + i_full[j+2] + i_full[j+3] + i_full[j+4]) * inv5
    for j in range(0, n_decimated, 5)  # Step by 5!
]
```

---

## 6. Control Systems: Automatic Gain Control (AGC)

Post-demodulation, arrays contain 64-bit Python floats. Audio soundcards require 16-bit packed PCM integer bytes ($-32768$ to $+32767$). 

Physical RF signal strengths vary by orders of magnitude (The Inverse-Square Law). If we apply a static scalar $K$ to the floats, a local station will overflow your integer boundaries (clipping), and a distant station will be $0.000001$ (silence).

We implement an **Automatic Gain Control (AGC)**: a non-linear closed-loop feedback system that tracks the output amplitude envelope and dynamically inverses the multiplier scale.

### Loop Dynamics: Attack and Release
A professional AGC utilizes differing exponential time constants to mimic human psychoacoustics:
*   **Attack Time ($\alpha_{att}$):** Fast ($\sim20$ ms). When a signal spikes, gain drops instantly to prevent math overflows.
*   **Release Time ($\alpha_{rel}$):** Slow ($\sim1.5$ sec). When a transmission ends, the gain crawls back up slowly to prevent loud, distracting "pumping" of background static.

```python
alp_att = 0.05    # Fast attack
alp_rel = 0.0001  # Slow release

for s in float_audio:
    scaled = s * static_user_gain * dynamic_agc_gain
    env = abs(scaled)
    
    # Non-linear Envelope Tracking Feedback Loop
    if env > 1.0:
        # Rapidly pull dynamic gain down toward ideal target (1.0 / env)
        dynamic_agc_gain += alp_att * (1.0 / env - dynamic_agc_gain)
    else:
        # Slowly let gain recover back toward maximum (1.0)
        dynamic_agc_gain += alp_rel * (1.0 - dynamic_agc_gain)
        
    # Hard clamp to prevent struct.pack int overflows
    clamped = max(-1.0, min(1.0, scaled))
    pulse_code = int(clamped * 32767.0)
```

---
# Part III: Modulation & Demodulation Calculus

How do we encode a 20 Hz – 20 kHz physical human voice onto a 100,000,000 Hz electromagnetic photon wave? 

A continuous unmodulated radio wave (the **Carrier**) is defined fundamentally by:
$$ s(t) = A_c \cos(2\pi f_c t + \phi_c) $$
We have three independent physical properties we can functionally manipulate (modulate) linearly or non-linearly with an audio baseband signal $m(t)$:
1.  **Amplitude ($A_c$)** $\rightarrow$ AM, SSB
2.  **Frequency ($f_c$)** $\rightarrow$ FM
3.  **Phase ($\phi_c$)** $\rightarrow$ PM (Phase Modulation, used in QPSK/WiFi)

---

## 7. Amplitude Modulation (AM)

**The Physics:**
In AM, the low-frequency audio signal $m(t)$ directly scales the transmission voltage of the high-frequency carrier.
$$ s_{AM}(t) = [A_c + m(t)] \cos(2\pi f_c t) $$

When analyzed formally in the frequency domain via Fourier Transform, multiplying a cosine carrier by an audio signal $m(t)$ results in the carrier spike ($f_c$) flanked symmetrically by two mirror-image "sidebands":
1.  Upper Sideband (USB): $f_c + m(f)$
2.  Lower Sideband (LSB): $f_c - m(f)$

**The Demodulator (Envelope Detection):**
When the receiver DSP downconverts this signal to $0$ Hz complex baseband, the carrier maps to exactly DC ($0$ Hz), and the identical sidebands stretch symmetrically outwards to $\pm f_{audio}$. 

Because the audio information is encoded purely and strictly in the *length* of the rotating complex I/Q vector, the phase angle of the vector contains absolutely no useful information. To demodulate, we simply calculate the Euclidean distance of the 2D complex vector from the origin using the Pythagorean theorem.

$$ \text{Audio Envelope} = ||z[n]|| = \sqrt{I[n]^2 + Q[n]^2} $$

*Note: The AM transmitter constantly broadcasts the empty Carrier Wave ($A_c$) so the envelope never crosses zero. Without a carrier, simple envelope detection fails (resulting in heavy distortion). However, this means our demodulated baseband audio will possess a massive constant DC bias (the carrier). We must explicitly pass the envelope through an IIR high-pass filter to strip the DC voltage before audio playback.*

```python
# AM Envelope Demodulator
env_full = [0.0] * n_pairs

for k in range(n_pairs):
    i_val = raw[k*2] - 127.5
    q_val = raw[k*2+1] - 127.5
    
    # 1. Non-linear Envelope Extraction (Pythagorean Theorem)
    env_full[k] = math.sqrt(i_val**2 + q_val**2)

# 2. Box-car Decimation ...

# 3. DC Removal (1-pole IIR high-pass filter)
# y[n] = x[n] - dc;  dc += alpha * (x[n] - dc)
dc = state.get('am_dc', 0.0)
alpha_dc = 0.001
for idx in range(len(audio)):
    dc += alpha_dc * (audio[idx] - dc)
    audio[idx] -= dc
```

---

## 8. Single Sideband (SSB)

**The Physics:**
Standard AM is an engineering disaster of inefficiency:
1. The constant carrier wave ($A_c$) contains strictly zero audio information but consumes $\sim 66\%$ of the transmitter's total power output.
2. The Upper and Lower sidebands contain the *exact same mirrored audio information*, wasting $50\%$ of the allocated physical RF bandwidth.

**SSB (Single Sideband)** solves this mathematically. The transmitter completely suppresses the carrier wave and utilizes steep quartz crystal hardware filters (or digital FIR Hilbert Transforms) to strip off one of the sidebands entirely. $100\%$ of transmitter power is now dedicated to a single, narrow $3$ kHz slice of spectrum. 

*   **USB (Upper Sideband):** Audio frequencies maintain their original orientation ($1 \text{ kHz tone} \rightarrow f_c + 1 \text{ kHz}$).
*   **LSB (Lower Sideband):** Audio frequencies are spectrally inverted ($1 \text{ kHz tone} \rightarrow f_c - 1 \text{ kHz}$).

**Generating SSB in Code (The Hilbert Transform):**
To generate a true SSB signal in software (e.g., our `mock_rtl_tcp.py` server), we cannot simply multiply an audio wave by a cosine. We must calculate the **Analytic Signal**: $a(t) = m(t) + j\cdot \hat{m}(t)$.
The imaginary component is the Hilbert Transform (a perfect 90° broadband phase shift) of the audio. By assigning the audio to the I channel and the Hilbert transform to the Q channel, the physical mathematics of complex modulation completely destructively interfere with and cancel out the unwanted sideband.

**The Demodulator (Analytic Spectral Folding):**
How do we demodulate a true SSB baseband signal without an envelope carrier? We rely on a profound mathematical artifact of complex baseband symmetry.

If the SDR is tuned *exactly* to where the suppressed carrier *used to exist*, the remaining sideband sits perfectly at baseband. 
*   USB sits purely at positive frequencies in the complex domain.
*   LSB sits purely at negative frequencies in the complex domain.

Recall that a complex signal ($I + jQ$) can represent both positive and negative frequencies on distinct rotational axes. But a **Real (1D) array** naturally, unavoidably forces positive and negative spectrums to sum together and merge symmetrically. 

By simply **dropping the Imaginary (Q) axis** and preserving only the Real (I) axis, the discrete positive or negative frequency shifts instantly collapse into a standard 1D audio sequence.

$$ \text{Audio Out} = \Re(z[n]) = I[n] $$

*No envelope detection, no phase discrimination, no complex BFO oscillators required! Given a true baseband SSB complex signal, the raw Real channel **is** the audio.*

```python
# SSB Baseband Demodulator
# For true baseband analytic SSB, Re(z) IS the audio equivalent.
i_full = [0.0] * n_pairs
for k in range(n_pairs):
    i_full[k] = raw[k * 2] - 127.5

# Followed instantaneously by box-car decimation and AGC.
# Dropping the Q channel natively folds the complex spectrum to 1D real audio.
```

---

## 9. Frequency Modulation (FM)

**The Physics:**
AM is highly susceptible to atmospheric static because physical phenomena like lighting strikes are literal amplitude spikes. **FM (Frequency Modulation)** achieves extreme high-fidelity transmission by holding the amplitude completely constant and encoding the audio $m(t)$ in the *continuous integral rate of change of the phase* (the instantaneous frequency) of the carrier.
$$ s_{FM}(t) = A_c \cos\left(2\pi f_c t + 2\pi k_f \int m(\tau) d\tau\right) $$

**The Demodulator (Polar Discrimination Calculus):**
Frequency is mathematically defined as the **first derivative of phase with respect to time**: $f(t) = \frac{1}{2\pi} \frac{d\theta}{dt}$.

Therefore, to extract the original pre-integral audio $m(t)$, we must:
1.  Calculate the absolute phase angle of each complex sample: $\theta = \arctan(Q/I)$
2.  Take the discrete time derivative: subtract the angle of the previous sample from the current sample: $\Delta \theta = \theta_n - \theta_{n-1}$

However, computing `math.atan2(Q, I)` twice per sample is extremely slow, and subtracting angles manually introduces brutal branch-logic bugs (e.g., $359° - 1° = +358°$, when the shortest path is actually $-2°$).

**The Polar Discriminator Bypass:**
We bypass raw trig functions by exploiting complex arithmetic rules. 
Multiplying a complex number $z_n$ by the **Complex Conjugate** of another $\overline{z_{n-1}}$ mathematically subtracts their angles inherently, instantly solving the wrap-around problem:
$$ \text{angle}(z_n \cdot \overline{z_{n-1}}) = \theta_n - \theta_{n-1} $$

We can execute the complex multiplication purely algebraically (using 4 floats), and only invoke `atan2` *once* on the final resulting vector.

```python
# WBFM Polar Discriminator (Instantaneous Phase Derivative)
for k in range(n_pairs):
    i_curr = raw[k*2] - 127.5
    q_curr = raw[k*2 + 1] - 127.5

    # Complex multiply: Z_curr * conj(Z_prev)
    # Z_prev = p_prev_i + j * p_prev_q -> conj(Z_prev) = p_prev_i - j * p_prev_q
    
    # Algebraically expanding the multiplication (Re and Im parts)
    p_i = i_curr * p_prev_i + q_curr * p_prev_q
    p_q = q_curr * p_prev_i - i_curr * p_prev_q

    # The resulting vector's angle IS the instantaneous discrete frequency derivative
    # Math.atan2 safely returns a bounded float between -π and +π (The Audio!)
    demodulated_audio[k] = math.atan2(p_q, p_i)
    
    # Save current complex sample for the next discrete derivative calculation
    p_prev_i, p_prev_q = i_curr, q_curr
```

*(Advanced Note: High-fidelity FM broadcasts utilize a mathematical "Pre-emphasis" high-pass filter at the radio station to boost high frequencies above the FM noise floor parabola. The receiver software must implement a matching "De-emphasis" 1-pole IIR low-pass filter (Time constant = 75µs in the US, 50µs in Europe) to flatten the audio response back to normal.)*

---

# Conclusion

Software Defined Radio strips away the archaic confines of soldering irons and analog drift, replacing them with perfect, immutable mathematics. By deeply understanding complex geometry, the rules of discrete recursive filtering, and differential calculus, you possess the theoretical power to write code that decodes passing NOAA weather satellites, analyzes global aircraft ADS-B trajectories, tracks AIS ships across oceans, or secures custom communication networks—relying solely on cheap hardware and the deterministic power of Python.

*End of Syllabus.* 📡🐍
