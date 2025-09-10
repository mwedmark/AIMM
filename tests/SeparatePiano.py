import numpy as np
import sounddevice as sd
from pynput import keyboard
import threading

# -----------------------------
# Piano synthesis
# -----------------------------
sample_rate = 44100
max_polyphony = 8
note_duration = 3.0  # seconds

active_notes = {}    # key -> {'wave': waveform, 'pos': pointer}
keys_down = set()    # track physical keys pressed
lock = threading.Lock()

def piano_note_sine(frequency, duration, velocity=1.0, sample_rate=44100):
    """Additive synthesis piano note with 6 partials and exponential decay"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    harmonics = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]
    waveform = sum(a * np.sin(2 * np.pi * frequency * (i+1) * t) for i, a in enumerate(harmonics))
    decay = np.exp(-3 * t)  # adjust decay speed
    waveform *= decay * velocity
    waveform /= np.max(np.abs(waveform))
    return waveform

# -----------------------------
# Keyboard mapping
# -----------------------------
key_note_map = {
    'a': 261.63,  # C4
    'w': 277.18,  # C#4
    's': 293.66,  # D4
    'e': 311.13,  # D#4
    'd': 329.63,  # E4
    'f': 349.23,  # F4
    't': 369.99,  # F#4
    'g': 392.00,  # G4
    'y': 415.30,  # G#4
    'h': 440.00,  # A4
    'u': 466.16,  # A#4
    'j': 493.88,  # B4
    'k': 523.25   # C5
}

# -----------------------------
# Audio callback
# -----------------------------
def audio_callback(outdata, frames, time, status):
    buffer = np.zeros(frames)
    to_delete = []

    with lock:
        for k, note in active_notes.items():
            start = note['pos']
            end = start + frames
            chunk = note['wave'][start:end]
            buffer[:len(chunk)] += chunk
            note['pos'] += len(chunk)
            if note['pos'] >= len(note['wave']):
                to_delete.append(k)
        for k in to_delete:
            del active_notes[k]

    # global clipping protection
    if np.max(np.abs(buffer)) > 1.0:
        buffer = buffer / np.max(np.abs(buffer))
    outdata[:] = buffer.reshape(-1,1)

# -----------------------------
# Keyboard handlers
# -----------------------------
def on_press(key):
    try:
        k = key.char
    except:
        return
    with lock:
        if k in key_note_map and k not in keys_down:
            keys_down.add(k)
            if k not in active_notes:
                freq = key_note_map[k]
                wave = piano_note_sine(freq, duration=note_duration, velocity=0.8)
                active_notes[k] = {'wave': wave, 'pos': 0}

def on_release(key):
    try:
        k = key.char
    except:
        return
    with lock:
        keys_down.discard(k)
        if k in active_notes:
            del active_notes[k]

# -----------------------------
# Run audio + keyboard
# -----------------------------
stream = sd.OutputStream(channels=1, callback=audio_callback,
                         samplerate=sample_rate, blocksize=1024)
stream.start()

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Keyboard piano ready! Press keys A-K to play. Ctrl+C to exit.")

try:
    while True:
        pass
except KeyboardInterrupt:
    stream.stop()
    listener.stop()
