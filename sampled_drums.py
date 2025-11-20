import numpy as np
import soundfile as sf
import os

class SampledDrums:
    def __init__(self, sample_dir="samples", sample_rate=44100):
        self.sample_dir = sample_dir
        self.sample_rate = sample_rate
        self.samples = {}
        self.load_samples()

    def load_samples(self):
        # Map General MIDI drum notes to samples
        self.note_map = {
            35: "Bass Drum 1.wav",   # Standard Kick (User preferred)
            36: "Bass Drum 1.wav",   # Standard Kick (User preferred)
            37: "Side Stick.wav",   # Side stick
            38: "Acoustic Snare.wav",  # Acoustic Snare
            39: "hand clap.wav",  # Clap
            40: "Electric Snare.wav",  # Electric Snare
            41: "Low Floor Tom.wav",  # 
            42: "Closed Hi-hat.wav",  # Closed Hi-hat
            43: "High Floor Tom.wav",  # 
            44: "Pedal Hi-hat.wav",   # Pedal Hi-hat
            45: "Low Tom.wav",   #
            46: "Open Hi-hat.wav",    # Open Hi-hat
            47: "Low-Mid tom.wav",    # Low Mid Tom
            48: "Hi Mid Tom.wav",    # Hi Tom
            49: "Crash Cymbal 1.wav",    # 
            50: "High Floor Tom.wav",    # 
            51: "Ride Cymbal 1.wav",    #
            52: "Chinese Cymbal.wav",    # 
            53: "Ride Bell.wav",    #
            54: "Tambourine.wav",    # 
            55: "Splash Cymbal.wav",    # 
            56: "Cowbell.wav",    #
            57: "Crash Cymbal 2.wav",    # 
            58: "Vibraslap.wav",    # 
            59: "Ride Cymbal 2.wav",    #
            60: "Hi Bongo.wav",
            61: "Low Bongo.wav",
            62: "Mute Hi Conga.wav",
            63: "Open Hi Conga.wav",
            64: "Low Conga.wav",
            65: "High Timbale.wav",
            66: "Low Timbale.wav",
            67: "High Agogo.wav",
            68: "Low Agogo.wav",
            69: "Cabasa.wav",
            70: "Maracas.wav",
            71: "Short Whistle.wav",
            72: "Long Whistle.wav",
            73: "Short Guiro.wav",
            74: "Long Guiro.wav",
            75: "Claves.wav",
            76: "Hi Wood Block.wav",
            77: "Low Wood Block.wav",
            78: "Mute Cuica.wav",
            79: "Open Cuica.wav",
            80: "Mute Triangle.wav",
            81: "Open Triangle.wav",
        }

        for note, filename in self.note_map.items():
            filepath = os.path.join(self.sample_dir, filename)
            if os.path.exists(filepath):
                data, sr = sf.read(filepath, dtype="float32")
                if sr != self.sample_rate:
                    raise ValueError(f"Sample {filename} has {sr}Hz, expected {self.sample_rate}Hz")
                self.samples[note] = data
            else:
                print(f"Warning: Missing sample {filename}")

    def trigger(self, note, velocity=100, pan=0.5):
        """Return stereo sample for a drum note"""
        if note not in self.samples:
            return np.zeros((1, 2), dtype=np.float32)

        mono_sample = self.samples[note]
        # Apply velocity scaling (0-127)
        volume = velocity / 127.0
        mono_sample = mono_sample * volume

        # Apply stereo panning
        left = mono_sample * np.cos(pan * np.pi / 2)
        right = mono_sample * np.sin(pan * np.pi / 2)
        stereo = np.stack([left, right], axis=1)
        return stereo
