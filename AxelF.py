import numpy as np
import wave
import winsound
import math
import random
from scipy.signal import butter, lfilter
from mido import MidiFile
from collections import defaultdict
from instruments import INSTRUMENT_PRESETS

# ================== Settings ==================
sample_rate = 44100
TEMPO_BPM = 110
BEATS_PER_SECOND = TEMPO_BPM / 60

DEFAULT_INSTRUMENT = 0  # fallback if no program_change

waveform_type_melody = 'triangle'
waveform_type_bass = 'triangle'


# Channel → pan value (0.0 = Left, 1.0 = Right)
channel_pan = {}

#Volume settings per channel
channel_volume = defaultdict(lambda: 1.0)  # Default volume is 1.0 (full)
channel_expression = defaultdict(lambda: 1.0)  # Default expression is 1.0 (velocity)

channel_pan[0] = 0.5  # Left
##channel_pan[1] = 0.8  # Right
##channel_pan[2] = 0.8  # Right
##channel_pan[3] = 0.8  # Right
##channel_pan[4] = 0.8  # Right
##channel_pan[9] = 0.5  # Center (drums)

# Default if not set: randomly assign pan per channel
def get_channel_pan(channel):
    if channel not in channel_pan:
        channel_pan[channel] = round(random.uniform(0.1, 0.9), 2)
    return channel_pan[channel]

def soft_limiter(signal, threshold=0.9):
    return np.tanh(signal / threshold) * threshold

def soft_limiter_agg(signal, drive=1.5):
      return np.tanh(signal * drive)

def normalize_rms(stereo, target_rms=0.2):
    rms = np.sqrt(np.mean(stereo**2))
    if rms > 0:
        stereo *= target_rms / rms
    return stereo

# ================== Note Frequency ==================
def get_note_frequency(note):
    if isinstance(note, int):  # MIDI note number
        A4_freq = 440.0
        A4_key = 69
        return A4_freq * 2 ** ((note - A4_key) / 12)
    
    if note == 'P':
        return 0.0
    
    # Reference: A4 = 440Hz, and A4 is the 49th key (starting from A0 = key 1)
    A4_freq = 440.0
    A4_key = 49
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    name = note[:-1]    
    octave = int(note[-1])
   
    if name not in note_names:
        raise ValueError(f"Invalid note name: {note}")

    key_number = note_names.index(name) + 12 * (octave + 1)
    freq = A4_freq * 2 ** ((key_number - A4_key) / 12)
    return round(freq, 2)
# ================== Note Conversion ==================
def note_len_to_beats(n):
    return 4 / n

# Normalize different event formats into a unified list of note events
def normalize_events(events):
    normalized = []
    current_beat = 0.0
    for item in events:
        if isinstance(item, tuple):
            notes, duration = item
            if isinstance(notes, str):
                notes = [notes]
            duration_beats = note_len_to_beats(duration)
            for note in notes:
                normalized.append({
                    'notes': [note],
                    'start_beats': current_beat,
                    'duration_beats': duration_beats,
                    'channel': -1
                })
            current_beat += duration_beats
        elif isinstance(item, dict):
            notes = item['notes'] if isinstance(item['notes'], list) else [item['notes']]
            start_beats = sum(note_len_to_beats(x) for x in item['start']) if isinstance(item['start'], list) else item['start']
            duration_beats = note_len_to_beats(item['length']) if isinstance(item['length'], int) else item['length']
            for note in notes:
                normalized.append({
                    'notes': [note],
                    'start_beats': start_beats,
                    'duration_beats': duration_beats,
                    'channel': -1
                })
    return normalized

def midi_to_events(midi_path, allowed_channels=None):
    mid = MidiFile(midi_path)
    events = []
    time = 0.0
    active_notes = {}
    channel_program = {}

    for msg in mid:
        time += msg.time
        if not msg.is_meta and hasattr(msg, 'channel'):
            if allowed_channels is not None and msg.channel not in allowed_channels:
                continue

        if msg.type == 'program_change':
            #if channel_program.get(msg.channel) != msg.program:
                #instrument_name = INSTRUMENT_PRESETS.get(msg.program, ("unknown",))[6]
                #print(f"Channel {msg.channel + 1} selected instrument: {instrument_name}")
            channel_program[msg.channel] = msg.program
        elif msg.type == 'control_change' and msg.control == 7:
            channel_volume[msg.channel] = msg.value / 127.0 
            #mess = f"Volume Set:{msg.channel+1}:{channel_volume[msg.channel]}"
            #print(mess)
            # MIDI volume is 0–127 → Normalize to 0.0–1.0
        elif msg.type == 'control_change' and msg.control == 11:
            channel_expression[msg.channel] = msg.value / 127.0
        elif msg.type == 'note_on' and msg.velocity > 0:
            active_notes[(msg.note, msg.channel)] = time
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.note, msg.channel)
            if key in active_notes:
                start = active_notes.pop(key)
                duration = time - start
                vol = channel_volume.get(msg.channel, 1.0)
                vol = max (vol, 0.05)
                velocity = msg.velocity / 127.0 #if msg.type == 'note_on' else 1.0  # fallback
                if(velocity == 0 and msg.type == 'note_on'):
                    velocity = 100
                #print (velocity)
                exp = channel_expression.get(msg.channel, 1.0)
                final_volume = vol * exp * velocity
                events.append({
                    'notes': [msg.note],
                    'start_beats': start * BEATS_PER_SECOND,
                    'duration_beats': duration * BEATS_PER_SECOND,
                    'channel': msg.channel,
                    'program': channel_program.get(msg.channel, DEFAULT_INSTRUMENT),
                    'volume': final_volume
                })
    return events


# ================== Waveform Generator ==================
def generate_wave(freq, duration, sr, wave_type='square', a_sec=0.02, d_sec=0.05, s_level=0.4, r_sec=0.08):
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, False)

    if freq == 0.0:
        return np.zeros(samples, dtype=np.float32)

    if wave_type == 'sine':
        wave = np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == 'triangle':
        wave = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
    elif wave_type == 'saw':
        wave = 2 * (t * freq % 1) - 1
    else:
        raise ValueError(f"Unknown wave_type: {wave_type}")

    # Envelope
    envelope = adsr_envelope(samples, sr, a_sec, d_sec, s_level, r_sec)
    shaped = wave * envelope

    # Optional lowpass to reduce aliasing (especially on saw/square)
    if wave_type in ['saw', 'square']:
        shaped = lowpass_filter(shaped, cutoff=8000, sr=sr)

    return shaped.astype(np.float32)

def generate_wave_steeldrum(freq, duration, sr, wave_type='triangle', a_sec=0.005, d_sec=0.12, s_level=0.3, r_sec=0.25):
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, False)

    base = np.sin(2 * np.pi * freq * t)  # Fundamental
    overtone1 = 0.2 * np.sin(2 * np.pi * freq * 2.01 * t)  # Slight detuned 2nd harmonic
    overtone2 = 0.1 * np.sin(2 * np.pi * freq * 3.9 * t)    # Almost 4th harmonic

    wave = base + overtone1 + overtone2
    wave /= np.max(np.abs(wave))  # normalize

    envelope = adsr_envelope(samples, sr, a_sec, d_sec, s_level, r_sec)
    return (wave * envelope).astype(np.float32)

# Optional: simulate sympathetic resonance from adjacent keys
def sympathetic_resonance(freq, t, strength=0.01):
    # Create quiet neighboring tones that share harmonics
    neighbor_freqs = [freq * r for r in [0.5, 0.75, 1.5, 2.0]]
    res_wave = np.zeros_like(t)
    for nf in neighbor_freqs:
        res_wave += np.sin(2 * np.pi * nf * t)
    return strength * res_wave


def generate_wave_piano(freq, duration, sr, a_sec=0.005, d_sec=0.2, s_level=0.4, r_sec=0.4):
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, False)

    # Gentle harmonic mix to avoid buildup
    wave = (
        0.8 * np.sin(2 * np.pi * freq * t) +              # Fundamental
        0.4 * np.sin(2 * np.pi * freq * 2 * t) +          # 2nd harmonic
        0.25 * np.sin(2 * np.pi * freq * 3 * t) +         # 3rd harmonic
        0.15 * np.sin(2 * np.pi * freq * 4.01 * t) +      # Slightly detuned 4th harmonic
        0.1 * np.sin(2 * np.pi * freq * 5.01 * t)         # Slightly detuned 5th harmonic
    )

    # Natural string damping
    damping = np.exp(-3.5 * t)
    wave *= damping
    # Normalize per note
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = wave / peak

    # ADSR
    envelope = adsr_envelope(samples, sr, a_sec, d_sec, s_level, r_sec)
    shaped = wave * envelope

    # Optional soft limiting (tame any slight overshoots)
    shaped = np.tanh(shaped * 1) #1.5 be default
    return shaped.astype(np.float32)

def generate_stereo_piano(freq, duration, sr):
    left = generate_wave_piano_better(freq * 0.997, duration, sr)
    right = generate_wave_piano_better(freq * 1.003, duration, sr)
    return np.stack([left, right], axis=-1)

def dynamic_lowpass(wave, sr, t):
    # Very crude time-dependent lowpass: cutoff drops over time
    out = np.zeros_like(wave)
    step = sr // 50
    for i in range(0, len(wave), step):
        t_pos = t[i] if i < len(t) else t[-1]
        cutoff = 8000 - 7000 * (t_pos / t[-1])  # drops from 8kHz to 1kHz
        b, a = butter(2, cutoff / (sr / 2), btype='low')
        chunk = wave[i:i+step]
        out[i:i+step] = lfilter(b, a, chunk)
    return out

def generate_wave_piano_better(freq, duration, sr, a_sec=0.005, d_sec=0.3, s_level=0.5, r_sec=0.5,velocity=0.8):
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, False)
    
    # Inharmonicity factor (approximated for acoustic grand)
    beta = 0.00015  # Slight stretch of upper partials
    
    harmonics = [  # (multiplier, amplitude)
        (1.0, 0.8),
        (2.005, 0.5),
        (3.01, 0.3),
        (4.02, 0.2),
        (5.03, 0.15),
        (6.06, 0.12),
        (7.1, 0.08),
        (8.15, 0.05),
    ]
    
    # Build waveform with stretched partials
    wave = np.zeros_like(t)
    for i, (mult, amp) in enumerate(harmonics):
        brightness = 1.0 + 0.5 * (velocity - 0.5) * (mult - 1)
        inharm_freq = freq * mult * (1 + beta * mult**2)
        wave += amp * brightness * np.sin(2 * np.pi * inharm_freq * t)
    
    #for i, (mult, amp) in enumerate(harmonics):
    #    inharm_freq = freq * mult * (1 + beta * mult**2)
    #    wave += amp * np.sin(2 * np.pi * inharm_freq * t)

    # Apply dynamic damping curve: high-freq components fade faster
    damping = np.exp(-2.8 * t) * (1.0 - 0.15 * t)
    wave *= damping
    wave += sympathetic_resonance(freq, t) # add extra resonance
    
    # Hammer noise transient: simulate initial percussive strike
    hammer_noise = 0.05 * np.random.randn(len(t))
    hammer_env = np.exp(-200 * t)
    wave += hammer_noise * hammer_env
    
    #attack_shape = np.tanh(20 * t)  # Sharper initial ramp
    #wave *= attack_shape

    # Normalize
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave /= peak

    # ADSR envelope
    envelope = adsr_envelope(samples, sr, a_sec, d_sec, s_level, r_sec)
    shaped = wave * envelope

    # Final soft saturation
    #shaped = dynamic_lowpass(shaped, sr, t)
    shaped = np.tanh(shaped * 1.2)
    
    return shaped.astype(np.float32)


def adsr_envelope(total_samples, sr, a_sec=0.02, d_sec=0.05, s_level=0.4, r_sec=0.08):
    a_len = int(a_sec * sr)
    d_len = int(d_sec * sr)
    r_len = int(r_sec * sr)
    s_len = total_samples - (a_len + d_len + r_len)
    s_len = max(0, s_len)

    attack = np.linspace(0, 1.0, a_len, False)
    decay = np.linspace(1.0, s_level, d_len, False)
    sustain = np.full(s_len, s_level)
    release = np.linspace(s_level, 0.0, r_len, False)

    envelope = np.concatenate([attack, decay, sustain, release])
    if len(envelope) < total_samples:
        envelope = np.pad(envelope, (0, total_samples - len(envelope)))
    else:
        envelope = envelope[:total_samples]
    return envelope

# ================== Synth Engine ==================
def group_events_by_channel(events):
    grouped = defaultdict(list)
    for event in events:
        channel = event.get('channel', 0)
        grouped[channel].append(event)
    return grouped

def synthesize_and_pan_channels(events_by_channel, note_freqs, sample_rate, seconds_per_beat):
    channel_tracks = []
    volume_scale = 1.0 / max(4, len(events_by_channel))  # avoid clipping when many channels

    for ch, ch_events in events_by_channel.items():
        waveform = waveform_type_melody
        audio, _ = synthesize_normalized_track(
            ch_events, note_freqs, waveform, sample_rate, seconds_per_beat
        )
        audio *= volume_scale
        pan = get_channel_pan(ch)
        left = audio * np.cos(pan * np.pi / 2)
        right = audio * np.sin(pan * np.pi / 2)
        stereo = np.stack([left, right], axis=1)
        channel_tracks.append(stereo)

    max_len = max(len(st) for st in channel_tracks)
    mix = np.zeros((max_len, 2), dtype=np.float32)
    for st in channel_tracks:
        st = np.pad(st, ((0, max_len - len(st)), (0, 0)))
        mix += st
    mix = np.clip(mix, -1.0, 1.0)
    return mix

def synthesize_normalized_track(events, note_freqs, default_waveform, sample_rate, seconds_per_beat):
    if not events:
        return np.zeros(1, dtype=np.float32), get_channel_pan(0)  # default mono + pan

    end_time = max(e['start_beats'] + e['duration_beats'] for e in events)
    total_samples = int(end_time * seconds_per_beat * sample_rate)
    track = np.zeros(total_samples, dtype=np.float32)

    used_channels = set()

    for event in events:
        program = event.get('program', DEFAULT_INSTRUMENT)
        channel = event.get('channel', 0)
        used_channels.add(channel)

        waveform, a, d, s, r, reverb_amount, name = INSTRUMENT_PRESETS.get(program,INSTRUMENT_PRESETS[DEFAULT_INSTRUMENT])
        for note in event['notes']:
            freq = note_freqs.get(note, get_note_frequency(note))
            if freq == 0.0:
                continue

            start_sample = int(event['start_beats'] * seconds_per_beat * sample_rate)
            duration = event['duration_beats'] * seconds_per_beat
            if program == 0:  # Acoustic Grand Piano
                wave = generate_wave_piano_better(freq, duration, sample_rate, a, d, s, r)
            elif program == 114: # Steel Drum
                wave = generate_wave_steeldrum(freq, duration, sample_rate, a, d, s, r)
            else:
                wave = generate_wave(freq, duration, sample_rate, waveform, a, d, s, r)
            
            wave = apply_fade_out(wave, sample_rate)

            end_sample = start_sample + len(wave)
            if end_sample > len(track):
                wave = wave[:len(track) - start_sample]

            volume = event.get('volume', 1.0)
            volume = max(0.0, min(1.0, volume)) # for safety
            voice_gain = 0.2  # lower gain per voice, adjust as needed
            track[start_sample:start_sample + len(wave)] += wave * volume * voice_gain

    # If multiple channels used, average their pans
    avg_pan = sum(get_channel_pan(ch) for ch in used_channels) / len(used_channels)
    return track, avg_pan

def apply_fade_out(wave, sr, fade_sec=0.01):
    fade_len = min(len(wave), int(sr * fade_sec))
    fade_curve = np.linspace(1.0, 0.0, fade_len)
    wave[-fade_len:] *= fade_curve
    return wave

def lowpass_filter(signal, cutoff=8000, sr=44100):
    b, a = butter(2, cutoff / (sr / 2), btype='low')
    return lfilter(b, a, signal)


# ================== Drum Generator ==================
def synth_drum_stereo(drum_type, duration, pan=0.5, sample_rate=44100):
    length = int(sample_rate * duration)
    t = np.linspace(0, duration, length, False)

    if drum_type == 'kick':
        envelope = np.exp(-40 * t)
        waveform = 0.8 * envelope * np.sin(2 * np.pi * 80 * t * (1 - 0.5 * t))
    elif drum_type == 'snare':
        noise = np.random.randn(length)
        envelope = np.exp(-20 * np.linspace(0, 1, length))
        waveform = 0.5 * noise * envelope
    elif drum_type == 'hat':
        noise = np.random.randn(length)
        envelope = np.exp(-50 * np.linspace(0, 1, length))
        waveform = 0.3 * noise * envelope
    else:
        waveform = np.zeros(length)

    # Apply panning to stereo
    left = waveform * np.cos(pan * np.pi / 2)
    right = waveform * np.sin(pan * np.pi / 2)
    return np.stack([left, right], axis=1).astype(np.float32)


def generate_drum_track_stereo(normalized_events, seconds_per_beat, sample_rate, drum_pan_map=None):
    if not normalized_events:
        return np.zeros((1, 2), dtype=np.float32)  # fallback for empty

    end_time = max(e['start_beats'] + e['duration_beats'] for e in normalized_events)
    total_samples = int(end_time * seconds_per_beat * sample_rate)
    stereo_track = np.zeros((total_samples, 2), dtype=np.float32)

    if drum_pan_map is None:
        drum_pan_map = {}

    for e in normalized_events:
        drum_type = e['notes'][0]
        start_sample = int(e['start_beats'] * seconds_per_beat * sample_rate)
        duration = e['duration_beats'] * seconds_per_beat

        # Get or assign panning
        key = (drum_type, e.get('channel', 9))  # default to channel 9 for drums
        pan = drum_pan_map.get(key, np.random.uniform(0.3, 0.7))
        drum_pan_map[key] = pan  # persist for consistency

        wave = synth_drum_stereo(drum_type, duration, pan, sample_rate)
        end_sample = start_sample + wave.shape[0]

        if end_sample > total_samples:
            wave = wave[:total_samples - start_sample]

        stereo_track[start_sample:start_sample + wave.shape[0]] += wave

    # Clip to prevent overflow
    stereo_track = np.clip(stereo_track, -1.0, 1.0)
    return stereo_track



# ================== Reverb ==================
#stereo, sample_rate,0.25,[50,113,179,271], wet=0.35, dry=0.8,lowpass_fc=5000)
def reverb_stereo(signal, sample_rate=44100, decay=0.25, delays_ms=[50,113,179,271], wet=0.35, dry=0.8, lowpass_fc=5000):
    def lowpass(sig, cutoff, fs):
        b, a = butter(1, cutoff / (fs / 2), btype='low')
        return lfilter(b, a, sig)

    wet_L = np.zeros_like(signal[:, 0], dtype=np.float32)
    wet_R = np.zeros_like(signal[:, 1], dtype=np.float32)

    for d_ms in delays_ms:
        d_samples = int(sample_rate * d_ms / 1000)
        echo_L = np.pad(signal[:, 0], (d_samples, 0))[:len(signal)] * decay
        echo_R = np.pad(signal[:, 1], (d_samples, 0))[:len(signal)] * decay
        wet_L += echo_L
        wet_R += echo_R

    wet_L = lowpass(wet_L, lowpass_fc, sample_rate)
    wet_R = lowpass(wet_R, lowpass_fc, sample_rate)

    out_L = dry * signal[:, 0] + wet * wet_L
    out_R = dry * signal[:, 1] + wet * wet_R

    stereo_out = np.stack([out_L, out_R], axis=1)
    stereo_out = np.clip(stereo_out, -1.0, 1.0)
    return stereo_out

# ================== Stereo Mixer ==================
def mix_voices_stereo(
    melody_stereo, drums_stereo,
    sr,
    melody_volume=1.0, melody_reverb=False,
    drums_volume=1.0, drums_reverb=False
):
    length = max(
        melody_stereo.shape[0] if melody_stereo is not None else 0,
        drums_stereo.shape[0] if drums_stereo is not None else 0
    )

    def pad_stereo(sig):
        if sig is None:
            return np.zeros((length, 2), dtype=np.float32)
        return np.pad(sig, ((0, length - sig.shape[0]), (0, 0)))

    m = pad_stereo(melody_stereo) * melody_volume
    d = pad_stereo(drums_stereo) * drums_volume

    if melody_reverb:
        m = reverb_stereo(m, sr)
    if drums_reverb:
        d = reverb_stereo(d, sr)

    mix = m + d
    mix = np.clip(mix, -1.0, 1.0)
    return mix.astype(np.float32)


# ================== Example Tracks ==================
melody_track = [
    ('C3', 4), ('D#3', 6), ('C3', 8), ('C3', 16), ('F3', 8), ('C3', 8), ('A#2', 8), ('C3', 4),
    ('G3', 6), ('C3', 8), ('C3', 16), ('G#3', 8), ('G3', 8), ('D#3', 8), ('C3', 8), ('G3', 8),
    ('C4', 8), ('C3', 16), ('A#2', 8), ('A#2', 16), ('G2', 8), ('D3', 8), ('C3', 2)
]

bass_track = [
    ('C0', 4), ('C1', 6), ('A#0', 8), ('A#1', 16), ('G0', 8), ('G1', 8), ('A#0', 8),
    ('C0', 4), ('C1', 6), ('P', 8), ('G0', 16), ('G1', 8), ('A#1', 8), ('C1', 8),
    ('G#0', 4), ('G#1', 6), ('A#0', 8), ('A#1', 16), ('G0', 8), ('A#0', 8), ('C0', 8),
    ('C1', 4), ('P', 4), ('P', 16), ('A#1', 16), ('G0', 8), ('F0', 8), ('D#0', 8)
]

drum_track = [
    ('kick', 8),('hat', 16),('hat', 16), ('snare', 6), ('kick', 8), ('kick', 16), ('kick', 8), ('snare', 8),('hat', 16),('hat', 16),
    ('kick', 8),('hat', 16), ('hat', 16),('snare', 6), ('kick', 8), ('kick', 16), ('kick', 8), ('snare', 4),
    ('kick', 8),('hat', 16),('hat', 16), ('snare', 6), ('kick', 8), ('kick', 16), ('kick', 8), ('snare', 8), ('hat', 16),('hat', 16),
    ('kick', 8),('hat', 16),('hat', 16), ('snare', 6), ('kick', 8), ('kick', 16), ('kick', 8), ('snare', 8), ('snare', 16), ('snare', 16)
]

##melody_track = [
##    ('C3', 4),
##    (['E3', 'G3'], 4),
##    {'notes': ['C4'], 'start': [4, 4], 'length': 8},
##    {'notes': ['F3', 'A3'], 'start': 2, 'length': 4}
##]
##
##bass_track = [
##    ('C1', 4),
##    ('F1', 4),
##    {'notes': ['G1'], 'start': [4], 'length': 4}
##]
##
##drum_track = [
##    ('kick', 4), ('hat', 8), ('snare', 4),
##    {'notes': 'hat', 'start': [2], 'length': 8},
##    {'notes': 'kick', 'start': [4], 'length': 4}
##]

#midi_events = midi_to_events("UnderTheSea.mid", allowed_channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
midi_events = midi_to_events("beethoven_opus10_2_format0.mid", allowed_channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

normalized_melody = midi_events
#normalized_melody = normalize_events(melody_track)

#normalized_bass = normalize_events(bass_track)

# Update note_freqs with both string and int note support
note_freqs = {}
all_events = normalized_melody # + normalized_bass
for event in all_events:
    for note in event['notes']:
        if note not in note_freqs:
            note_freqs[note] = get_note_frequency(note)
            
melody_events_by_channel = group_events_by_channel(normalized_melody)
melody_stereo = synthesize_and_pan_channels(melody_events_by_channel, note_freqs, sample_rate, 1 / BEATS_PER_SECOND)
#generate_stereo_piano(melody_events_by_channel, note_freqs, sample_rate)

normalized_drums = normalize_events(drum_track)
drum_stereo = generate_drum_track_stereo(normalized_drums, 1 / BEATS_PER_SECOND, sample_rate)

# Pad to same length and mix
#max_len = max(len(melody_stereo), len(drum_stereo))
#melody_stereo = np.pad(melody_stereo, (0, max_len - len(melody_stereo)))
#drum_stereo = np.pad(drum_stereo, (0, max_len - len(drum_stereo)))

# Mix stereo
stereo = mix_voices_stereo(
    melody_stereo, drum_stereo, sr=sample_rate,
    melody_volume=0.3, melody_reverb=True,
    drums_volume=0, drums_reverb=False
)

# ================== Final WAV Output ==================
# Normalize
max_val = np.max(np.abs(stereo))
if max_val > 1.0:
    stereo = stereo / max_val

# Apply soft limiter and normalize RMS
stereo = soft_limiter_agg(stereo)
stereo = normalize_rms(stereo)

stereo = np.clip(stereo, -1.0, 1.0)

# Save to WAV
filename = "polyphonic_output.wav"
with wave.open(filename, 'w') as wav_file:
    wav_file.setnchannels(2)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    int_samples = (stereo * 32767).astype(np.int16)
    wav_file.writeframes(int_samples.tobytes())

# Play
winsound.PlaySound(filename, winsound.SND_FILENAME)
