#!/usr/bin/env python3
"""Synthesise an original 3-beep alert sound from scratch (numpy only, no sampled audio).

Variant A (default): E6 3-beep chime with a soft ringing tail, peak about -2.6 dBFS before MP3 encoding (about -3 dBFS after).
Variant B (softer): D6, slower attack, fewer harmonics, shorter/quieter tail, peak -6 dBFS.
Outputs mono 44.1 kHz WAV + MP3 (libmp3lame 128 kbps).
Usage: python3 make_alert.py <outdir_for_A> <outdir_for_B_and_wavs>
"""
import sys, subprocess, wave
import numpy as np

SR = 44100

def beep_env(n, attack, sustain, decay):
    """0..n samples: linear attack, flat sustain, exponential-ish decay to ~0."""
    t = np.arange(n) / SR
    env = np.ones(n)
    env[t < attack] = t[t < attack] / attack
    td = t - (attack + sustain)
    m = td > 0
    env[m] = np.exp(-5.0 * td[m] / decay) * (1 - td[m] / (decay * 1.6)).clip(0, 1)
    return env

def synth(f0, beeps, spacing, attack, sustain, decay, h2, h3, tail_db, tail_tau, detune, total, peak_db, gap_tau=0.12, gap_gain=1.0):
    n = int(total * SR)
    y = np.zeros(n)
    def tone(t, f, a2, a3):
        ph = 2 * np.pi * f * t
        return np.sin(ph) + a2 * np.sin(2 * ph + 0.6) + a3 * np.sin(3 * ph + 1.1)
    plen = int((attack + sustain + decay * 1.6) * SR)
    t = np.arange(plen) / SR
    for k in range(beeps):
        s = int(k * spacing * SR)
        y[s:s + plen] += (tone(t, f0, h2, h3) * beep_env(plen, attack, sustain, decay))[:n - s]
    # soft ringing after every beep (fills the gaps a little, like a small bell/room): two slightly
    # detuned decaying sines; the ring after the last beep is the longest "tail".
    amp = 10 ** (tail_db / 20)
    for k in range(beeps):
        last = (k == beeps - 1)
        t0 = k * spacing + attack + sustain
        tt = np.arange(n) / SR - t0
        m = tt > 0
        tau = tail_tau if last else gap_tau
        a = amp if last else amp * gap_gain
        ring = np.zeros(n)
        ring[m] = np.exp(-tt[m] / tau) * (np.sin(2 * np.pi * f0 * tt[m]) + 0.6 * np.sin(2 * np.pi * f0 * (1 + detune) * tt[m] + 0.9))
        ring[m] *= (1 - np.exp(-tt[m] / 0.015))        # smooth start
        y += a * ring
    # fades to avoid clicks
    fi, fo = int(0.004 * SR), int(0.06 * SR)
    y[:fi] *= np.linspace(0, 1, fi)
    y[-fo:] *= np.linspace(1, 0, fo)
    y *= 10 ** (peak_db / 20) / np.abs(y).max()
    return y

VARIANTS = {
    # name: params
    'A': dict(f0=1320.0, beeps=3, spacing=0.135, attack=0.006, sustain=0.055, decay=0.070, h2=0.10, h3=0.10,
              tail_db=-14.0, tail_tau=0.36, detune=0.012, total=1.45, peak_db=-2.6, gap_tau=0.14, gap_gain=0.9),
    'B': dict(f0=1174.7, beeps=3, spacing=0.150, attack=0.014, sustain=0.050, decay=0.065, h2=0.03, h3=0.02,
              tail_db=-18.0, tail_tau=0.26, detune=0.010, total=1.35, peak_db=-6.0, gap_tau=0.10, gap_gain=0.8),
}

def write(y, base):
    pcm = (np.clip(y, -1, 1) * 32767).astype('<i2')
    with wave.open(base + '.wav', 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-i', base + '.wav', '-ac', '1', '-c:a', 'libmp3lame', '-b:a', '128k', base + '.mp3'])

if __name__ == '__main__':
    outA, outB = sys.argv[1], sys.argv[2]
    write(synth(**VARIANTS['A']), outA.rstrip('/') + '/alert')
    write(synth(**VARIANTS['B']), outB.rstrip('/') + '/alert_B')
