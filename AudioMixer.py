import wave
import numpy as np
from scipy.signal import lfilter, butter


class AudioMixer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.tracks = []  # List of (track, volume, pan, apply_reverb) tuples
        
    def add_track(self, track, volume=1.0, pan=0.5, apply_reverb=False):
        """Add a track to the mixer with specified properties"""
        self.tracks.append((track, volume, pan, apply_reverb))
        
    def apply_stereo_pan(self, mono_track, pan):
        """Convert mono track to stereo with panning"""
        left = mono_track * np.cos(pan * np.pi / 2)
        right = mono_track * np.sin(pan * np.pi / 2)
        return np.stack([left, right], axis=1)
        
    def reverb_stereo(self, signal, decay=0.25, delays_ms=[50,113,179,271], wet=0.35, dry=0.8, lowpass_fc=5000):
        """Apply stereo reverb effect"""
        def lowpass(sig, cutoff, fs):
            b, a = butter(1, cutoff / (fs / 2), btype='low')
            return lfilter(b, a, sig)
            
        wet_L = np.zeros_like(signal[:, 0], dtype=np.float32)
        wet_R = np.zeros_like(signal[:, 1], dtype=np.float32)
        
        for d_ms in delays_ms:
            d_samples = int(self.sample_rate * d_ms / 1000)
            echo_L = np.pad(signal[:, 0], (d_samples, 0))[:len(signal)] * decay
            echo_R = np.pad(signal[:, 1], (d_samples, 0))[:len(signal)] * decay
            wet_L += echo_L
            wet_R += echo_R
            
        wet_L = lowpass(wet_L, lowpass_fc, self.sample_rate)
        wet_R = lowpass(wet_R, lowpass_fc, self.sample_rate)
        
        out_L = dry * signal[:, 0] + wet * wet_L
        out_R = dry * signal[:, 1] + wet * wet_R
        
        stereo_out = np.stack([out_L, out_R], axis=1)
        stereo_out = np.clip(stereo_out, -1.0, 1.0)
        return stereo_out
        
    def soft_limiter(self, signal, drive=1.5):
        """Apply soft limiting to prevent clipping"""
        return np.tanh(signal * drive)
        
    def normalize_rms(self, signal, target_rms=0.2):
        """Normalize signal to target RMS level"""
        rms = np.sqrt(np.mean(signal**2))
        if rms > 0:
            signal *= target_rms / rms
        return signal
        
    def mix(self):
        """Mix all tracks and apply effects"""
        if not self.tracks:
            return np.zeros((1, 2), dtype=np.float32)
            
        # Find max length of all tracks
        max_length = max(track.shape[0] if len(track.shape) > 1 else len(track) 
                          for track, _, _, _ in self.tracks)
        
        # Initialize output stereo mix
        mix = np.zeros((max_length, 2), dtype=np.float32)
        
        # Process each track
        for track, volume, pan, apply_reverb in self.tracks:
            # Convert mono to stereo if needed
            if len(track.shape) == 1:
                stereo_track = self.apply_stereo_pan(track, pan)
            else:
                stereo_track = track
                
            # Pad to max length
            padded = np.zeros((max_length, 2), dtype=np.float32)
            padded[:len(stereo_track)] = stereo_track[:max_length]
            
            # Apply reverb if requested
            if apply_reverb:
                padded = self.reverb_stereo(padded)
                
            # Apply volume and add to mix
            mix += padded * volume
            
        # Apply processing chain
        mix = np.clip(mix, -1.0, 1.0)
        mix = self.soft_limiter(mix)
        mix = self.normalize_rms(mix)
        mix = np.clip(mix, -1.0, 1.0)
        
        return mix
        
    def render_to_wav(self, output_path):
        """Render mixed audio to WAV file"""
        final_mix = self.mix()
        
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            int_samples = (final_mix * 32767).astype(np.int16)
            wav_file.writeframes(int_samples.tobytes())
            
        return output_path