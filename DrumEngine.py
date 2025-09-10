import numpy as np
import wave

class DrumEngine:
    def __init__(self, sample_rate=44100, use_samples=False, samples_dir="drum_samples"):
        self.sample_rate = sample_rate
        self.use_samples = use_samples
        self.samples_dir = samples_dir
        self.samples = {}
        
        # Frequencies and parameters for synthetic drums
        self.drum_params = {
            'kick': {
                'freq': 60.0,
                'decay': 0.6,
                'volume': 1.0
            },
            'snare': {
                'freq': 150.0,
                'decay': 0.3,
                'volume': 0.8
            },
            'hat': {
                'freq': 300.0,
                'decay': 0.1,
                'volume': 0.6
            }
        }
        
        # Load samples if in sample mode
        if self.use_samples:
            self.load_samples()
    
    def load_samples(self):
        """Load drum samples from the samples directory"""
        import os
        for drum_type in ['kick', 'snare', 'hat']:
            try:
                sample_path = os.path.join(self.samples_dir, f"{drum_type}.wav")
                if os.path.exists(sample_path):
                    with wave.open(sample_path, 'rb') as wf:
                        framerate = wf.getframerate()
                        n_frames = wf.getnframes()
                        data = wf.readframes(n_frames)
                        sample = np.frombuffer(data, dtype=np.int16) / 32767.0
                        self.samples[drum_type] = sample.astype(np.float32)
            except Exception as e:
                print(f"Error loading sample {drum_type}: {e}")
    
    def get_drum_sound(self, drum_type, duration):
        """Generate or retrieve a drum sound based on mode"""
        if self.use_samples and drum_type in self.samples:
            return self._get_sample(drum_type, duration)
        else:
            return self._synthesize_drum(drum_type, duration)
    
    def _get_sample(self, drum_type, duration):
        """Retrieve and format a pre-loaded sample"""
        if drum_type not in self.samples:
            return self._synthesize_drum(drum_type, duration)
            
        sample = self.samples[drum_type]
        target_samples = int(duration * self.sample_rate)
        
        if len(sample) > target_samples:
            return sample[:target_samples]
        else:
            return np.pad(sample, (0, target_samples - len(sample)))
    
    def _synthesize_drum(self, drum_type, duration):
        """Synthesize a drum sound based on type"""
        from scipy.signal import butter, lfilter
        
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        if drum_type == 'kick':
            # Bass drum: sine wave with pitch envelope
            freq_start = 150
            freq_end = 50
            freq = freq_start * np.exp(-t * 20) + freq_end
            envelope = np.exp(-t * 20)
            waveform = np.sin(2 * np.pi * freq * t) * envelope
            
        elif drum_type == 'snare':
            # Snare: noise + tone
            noise = np.random.uniform(-1, 1, samples)
            tone = np.sin(2 * np.pi * 180 * t)
            decay = np.exp(-t * 15)
            waveform = (0.7 * noise + 0.3 * tone) * decay
            
        elif drum_type == 'hat':
            # Hi-hat: filtered noise with fast decay
            noise = np.random.uniform(-1, 1, samples)
            decay = np.exp(-t * 40)
            waveform = noise * decay
            # High-pass filter
            b, a = butter(4, 2000 / (self.sample_rate / 2), 'high')
            waveform = lfilter(b, a, waveform)
            
        else:
            # Default
            noise = np.random.uniform(-0.5, 0.5, samples)
            decay = np.exp(-t * 10)
            waveform = noise * decay
        
        # Apply volume from parameters
        volume = self.drum_params.get(drum_type, {}).get('volume', 0.8)
        return (waveform * volume).astype(np.float32)