import numpy as np
from scipy.signal import butter, lfilter, sawtooth, square

class MidiInstruments:

    @staticmethod
    def lowpass_filter(wave, sr, cutoff=3000):
        b, a = butter(2, cutoff / (sr / 2), btype='low')
        return lfilter(b, a, wave)

    @staticmethod
    def adsr_envelope(samples, sr, a_sec, d_sec, s_level, r_sec):
        a_samples = int(sr * a_sec)
        d_samples = int(sr * d_sec)
        r_samples = int(sr * r_sec)
        s_samples = samples - (a_samples + d_samples + r_samples)
        if s_samples < 0:
            s_samples = 0

        a_env = np.linspace(0, 1.0, a_samples, False)
        d_env = np.linspace(1.0, s_level, d_samples, False)
        s_env = np.full(s_samples, s_level)
        r_env = np.linspace(s_level, 0, r_samples, False)
        return np.concatenate((a_env, d_env, s_env, r_env))[:samples]
    
    @staticmethod
    def generate_wave_steeldrum(freq, duration, sr, wave_type='triangle', a_sec=0.005, d_sec=0.12, s_level=0.3, r_sec=0.25):
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, False)

        base = np.sin(2 * np.pi * freq * t)  # Fundamental
        overtone1 = 0.2 * np.sin(2 * np.pi * freq * 2.01 * t)  # Slight detuned 2nd harmonic
        overtone2 = 0.1 * np.sin(2 * np.pi * freq * 3.9 * t)    # Almost 4th harmonic

        wave = base + overtone1 + overtone2
        wave /= np.max(np.abs(wave))  # normalize

        envelope = MidiInstruments.adsr_envelope(samples, sr, a_sec, d_sec, s_level, r_sec)
        return (wave * envelope).astype(np.float32)

    @staticmethod
    def generate_wave_piano(freq, duration, sr, a_sec=0.005, d_sec=0.2, s_level=0.4, r_sec=0.4):
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, False)

        wave = (
            0.8 * np.sin(2 * np.pi * freq * t) +
            0.4 * np.sin(2 * np.pi * freq * 2 * t) +
            0.25 * np.sin(2 * np.pi * freq * 3 * t) +
            0.15 * np.sin(2 * np.pi * freq * 4.01 * t) +
            0.1 * np.sin(2 * np.pi * freq * 5.01 * t)
        )

        damping = np.exp(-3.5 * t)
        wave *= damping

        peak = np.max(np.abs(wave))
        if peak > 0:
            wave = wave / peak

        envelope = MidiInstruments.adsr_envelope(samples, sr, a_sec, d_sec, s_level, r_sec)
        shaped = wave * envelope

        shaped = np.tanh(shaped * 1.0)  # Optional soft limiter
        return shaped.astype(np.float32)

    @staticmethod
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

    @staticmethod
    def generate_wave_nylon_guitar(freq, duration, sr, a_sec=0.003, d_sec=0.15, s_level=0.6, r_sec=0.6):
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, False)

        # Base pluck: impulse with slight noise and filtered delay (Karplus-Strong-inspired)
        pluck_noise = np.random.uniform(-1, 1, samples) * np.exp(-20 * t)
        base_wave = np.zeros_like(t)
        
        delay_samples = int(sr / freq)
        if delay_samples < 2: delay_samples = 2
        buf = np.random.randn(delay_samples) * 0.6

        # Simple looped feedback delay (damped comb filter)
        for i in range(samples):
            base_wave[i] = buf[i % delay_samples]
            avg = 0.5 * (buf[i % delay_samples] + buf[(i - 1) % delay_samples])
            buf[i % delay_samples] = avg * 0.998  # decay factor

        # Blend pluck noise for brightness
        wave = 0.8 * base_wave + 0.2 * pluck_noise

        # Add light harmonic enrichment (not too sharp)
        wave += (
            0.15 * np.sin(2 * np.pi * freq * 2 * t) +
            0.07 * np.sin(2 * np.pi * freq * 3.02 * t)
        )

        # Body resonance simulation (slight amplitude modulation)
        body_lfo = 1 + 0.03 * np.sin(2 * np.pi * 2 * t)
        wave *= body_lfo

        # Lowpass to simulate soft nylon string tone
        wave = MidiInstruments.lowpass_filter(wave, sr, cutoff=3200)

        # Normalize
        peak = np.max(np.abs(wave))
        if peak > 0:
            wave = wave / peak

        # ADSR envelope
        envelope = MidiInstruments.adsr_envelope(samples, sr, a_sec, d_sec, s_level, r_sec)
        shaped = wave * envelope

        # Soft saturation for warmth
        shaped = np.tanh(shaped * 1.1)

        return shaped.astype(np.float32)

    @staticmethod
    def generate_karplus_strong_guitar(freq, duration, sr, velocity=1.0):
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, False)

        # Karplus-Strong initialization
        delay_samples = int(sr / freq)
        noise = (2 * np.random.rand(delay_samples) - 1) * np.hanning(delay_samples)
        buf = np.copy(noise)

        output = np.zeros(samples)
        idx = 0

        # Setup a lowpass filter for string damping
        b, a = butter(1, 0.4 + 0.5 * velocity, btype='low')

        prev = 0.0
        decay = 0.995 + 0.002 * velocity  # longer decay for higher velocity

        for i in range(samples):
            # Linear interpolation fractional delay (simple approximation)
            val = buf[idx]
            avg = 0.5 * (val + prev)
            filtered = lfilter(b, a, [avg])[0]
            buf[idx] = filtered * decay
            prev = val

            output[i] = val
            idx += 1
            if idx >= delay_samples:
                idx = 0

        # Add sympathetic partials (simulating string body resonance)
        output += 0.1 * np.sin(2 * np.pi * freq * 2 * t)
        output += 0.05 * np.sin(2 * np.pi * freq * 3.01 * t)
        output += 0.03 * np.sin(2 * np.pi * freq * 5.02 * t)

        # Normalize to prevent clipping
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak

        # ADSR envelope for note shaping
        env = MidiInstruments.adsr_envelope(samples, sr, a_sec=0.01, d_sec=0.4, s_level=0.5, r_sec=0.8)
        output *= env

        # Apply velocity amplitude
        output *= velocity

        # Final lowpass to soften brightness
        output = MidiInstruments.lowpass_filter(output, sr, cutoff=3200)

        # Warm saturation
        output = np.tanh(output * 1.2)

        return output.astype(np.float32)

    @staticmethod
    def generate_church_organ(freq, duration, sr=44100, tremulant=True):

        if duration <= 0:
            print(f"[Warning] Skipped note: freq={freq} with non-positive duration {duration}")

        samples = int(sr * duration)
        if samples <= 0:
            return np.zeros(0)  # or small silence array
        
        t = np.linspace(0, duration, samples, False)

        # Harmonic layering
        detune_ratio = 2 ** (2 / 1200)
        f_base = freq
        f_4 = freq * 2
        f_16 = freq / 2
        f_detune = freq * detune_ratio

        wave = (
            0.5 * np.sin(2 * np.pi * f_base * t) +
            0.2 * np.sin(2 * np.pi * f_4 * t) +
            0.15 * np.sin(2 * np.pi * f_16 * t) +
            0.1 * square(2 * np.pi * f_base * t) +
            0.05 * sawtooth(2 * np.pi * f_detune * t)
        )

        # Tremulant: amplitude LFO
        if tremulant:
            lfo_rate = 5.0  # Hz
            lfo_depth = 0.12
            lfo = 1.0 + lfo_depth * np.sin(2 * np.pi * lfo_rate * t)
            wave *= lfo

        # Normalize
        wave /= np.max(np.abs(wave) + 1e-6)

        # Envelope
        envelope = MidiInstruments.adsr_envelope(samples, sr)
        wave *= envelope

        # Lowpass filtering
        wave = MidiInstruments.lowpass_filter(wave, sr, 4200)

        # Gentle saturation
        wave = np.tanh(wave * 1.2)

        return wave.astype(np.float32)
