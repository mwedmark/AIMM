from builtins import int, ValueError, set, sum, len, min
import numpy as np
from scipy.signal import lfilter, butter
import MidiInstrumentsSpec as INSTRUMENT_PRESETS
import DrumEngine

class Synthesizer:
    def __init__(self, sample_rate=44100, use_sample_drums=False):
        self.sample_rate = sample_rate
        self.drum_engine = DrumEngine.DrumEngine(sample_rate, use_sample_drums)
        self.instruments = INSTRUMENT_PRESETS  # Reference to the existing instruments class
        
    def generate_wave(self, freq, duration, wave_type='square', a_sec=0.02, d_sec=0.05, s_level=0.4, r_sec=0.08):
        samples = int(self.sample_rate * duration)
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
        envelope = self.adsr_envelope(samples, a_sec, d_sec, s_level, r_sec)
        shaped = wave * envelope
        
        # Optional lowpass to reduce aliasing (especially on saw/square)
        if wave_type in ['saw', 'square']:
            shaped = self.lowpass_filter(shaped, cutoff=8000)
            
        return shaped.astype(np.float32)
        
    def adsr_envelope(self, total_samples, a_sec=0.02, d_sec=0.05, s_level=0.4, r_sec=0.08):
        a_len = int(a_sec * self.sample_rate)
        d_len = int(d_sec * self.sample_rate)
        r_len = int(r_sec * self.sample_rate)
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
        
    def lowpass_filter(self, signal, cutoff=8000):
        b, a = butter(2, cutoff / (self.sample_rate / 2), btype='low')
        return lfilter(b, a, signal)

    def synthesize_track(self, events, note_freqs, seconds_per_beat, instrument_presets):
        if not events:
            return np.zeros(1, dtype=np.float32), 0.5  # default mono + pan

        end_time = max(e['start_beats'] + e['duration_beats'] for e in events)
        total_samples = int(end_time * seconds_per_beat * self.sample_rate)
        track = np.zeros(total_samples, dtype=np.float32)

        used_channels = set()
        default_params = ('square', 0.02, 0.05, 0.4, 0.08, 0.8, 'Default')

        for event in events:
            program = event.get('program', 0)
            channel = event.get('channel', 0)
            used_channels.add(channel)

            # Special handling for drum channel (9)
            is_drum = channel == 9 or 'midinote' in event

            if not is_drum:
                # Get instrument parameters
                try:
                    if hasattr(instrument_presets, f'INSTRUMENT_{program}'):
                        instrument_params = getattr(instrument_presets, f'INSTRUMENT_{program}')
                    else:
                        instrument_params = getattr(instrument_presets, 'INSTRUMENT_0', default_params)
                except (AttributeError, TypeError):
                    instrument_params = default_params

                waveform, a, d, s, r, mix_vol, name = instrument_params

            for note in event['notes']:
                start_sample = int(event['start_beats'] * seconds_per_beat * self.sample_rate)
                duration = event['duration_beats'] * seconds_per_beat

                if is_drum:
                    # Use drum engine for drum sounds
                    wave = self.drum_engine.get_drum_sound(note, duration)
                else:
                    # Regular instrument synthesis
                    freq = note_freqs.get(note, 0)
                    if freq == 0.0:
                        continue
                    wave = self.generate_wave(freq, duration, waveform, a, d, s, r)
                    wave = self.apply_fade_out(wave)

                end_sample = start_sample + len(wave)
                if end_sample > len(track):
                    wave = wave[:len(track) - start_sample]

                volume = event.get('volume', 1.0)
                volume = max(0.0, min(1.0, volume))  # for safety
                voice_gain = 0.2 if not is_drum else 0.8  # higher gain for drums

                # Apply volume (use mix_vol for non-drums)
                mix_factor = mix_vol if not is_drum else 1.0
                track[start_sample:start_sample + len(wave)] += wave * volume * voice_gain * mix_factor

        # Calculate average pan
        avg_pan = 0.5  # default center
        if used_channels:
            # Get a reference to the MIDIProcessor for pan calculation
            from inspect import currentframe
            context = currentframe().f_back.f_globals
            midi_processor = context.get('midi_processor')
            if midi_processor and hasattr(midi_processor, 'get_channel_pan'):
                avg_pan = sum(midi_processor.get_channel_pan(ch) for ch in used_channels) / len(used_channels)

        return track, avg_pan
    
    def apply_fade_out(self, wave, fade_sec=0.01):
        fade_len = min(len(wave), int(self.sample_rate * fade_sec))
        fade_curve = np.linspace(1.0, 0.0, fade_len)
        wave[-fade_len:] *= fade_curve
        return wave