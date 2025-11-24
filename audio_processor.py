"""
AudioProcessor - Handles all audio signal processing operations
"""
import numpy as np
from scipy.signal import butter, lfilter


class AudioProcessor:
    """Encapsulates audio signal processing functions including limiting, 
    normalization, filtering, ADSR envelopes, and reverb effects."""
    
    @staticmethod
    def soft_limiter(signal, threshold=0.9):
        """Apply soft limiting to prevent clipping.
        
        Args:
            signal: Input audio signal
            threshold: Threshold level for limiting (default 0.9)
            
        Returns:
            Limited signal
        """
        return np.tanh(signal / threshold) * threshold
    
    @staticmethod
    def soft_limiter_agg(signal, drive=1.5):
        """Apply aggressive soft limiting using tanh.
        
        Args:
            signal: Input audio signal
            drive: Drive amount for saturation (default 1.5)
            
        Returns:
            Limited signal
        """
        return np.tanh(signal * drive)
    
    @staticmethod
    def normalize_rms(stereo, target_rms=0.2):
        """Normalize audio to target RMS level.
        
        Args:
            stereo: Input stereo signal
            target_rms: Target RMS level (default 0.2)
            
        Returns:
            Normalized signal
        """
        rms = np.sqrt(np.mean(stereo**2))
        if rms > 0:
            stereo *= target_rms / rms
        return stereo
    
    @staticmethod
    def adsr_envelope(total_samples, sr, a_sec=0.02, d_sec=0.05, s_level=0.4, r_sec=0.08):
        """Generate ADSR (Attack, Decay, Sustain, Release) envelope.
        
        Args:
            total_samples: Total number of samples
            sr: Sample rate
            a_sec: Attack time in seconds
            d_sec: Decay time in seconds
            s_level: Sustain level (0.0 to 1.0)
            r_sec: Release time in seconds
            
        Returns:
            Envelope array
        """
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
    def apply_fade_out(wave, sr, fade_sec=0.01):
        """Apply fade out to the end of a waveform.
        
        Args:
            wave: Input waveform
            sr: Sample rate
            fade_sec: Fade duration in seconds (default 0.01)
            
        Returns:
            Waveform with fade out applied
        """
        fade_len = min(len(wave), int(sr * fade_sec))
        fade_curve = np.linspace(1.0, 0.0, fade_len)
        wave[-fade_len:] *= fade_curve
        return wave
    
    @staticmethod
    def lowpass_filter(signal, cutoff=8000, sr=44100):
        """Apply lowpass filter to reduce high frequencies.
        
        Args:
            signal: Input signal
            cutoff: Cutoff frequency in Hz (default 8000)
            sr: Sample rate (default 44100)
            
        Returns:
            Filtered signal
        """
        b, a = butter(2, cutoff / (sr / 2), btype='low')
        return lfilter(b, a, signal)
    
    @staticmethod
    def reverb_stereo(signal, sample_rate=44100, decay=0.25, delays_ms=[50, 113, 179, 271], 
                     wet=0.35, dry=0.8, lowpass_fc=5000):
        """Apply stereo reverb effect using multiple delays.
        
        Args:
            signal: Input stereo signal (N x 2)
            sample_rate: Sample rate (default 44100)
            decay: Decay factor for echoes (default 0.25)
            delays_ms: List of delay times in milliseconds
            wet: Wet (reverb) level (default 0.35)
            dry: Dry (original) level (default 0.8)
            lowpass_fc: Lowpass filter cutoff frequency (default 5000)
            
        Returns:
            Stereo signal with reverb applied
        """
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
    
    @staticmethod
    def chorus(signal, sample_rate=44100, num_voices=3, delay_ms=30, depth=0.8, 
               rate_hz=0.5, wet=0.5, dry=1.0):
        """Apply chorus effect to create a richer, more spacious sound.
        
        Args:
            signal: Input stereo signal (N x 2)
            sample_rate: Sample rate (default 44100)
            num_voices: Number of chorus voices (default 3)
            delay_ms: Base delay time in milliseconds (default 30)
            depth: Modulation depth in milliseconds (default 0.8)
            rate_hz: LFO rate in Hz (default 0.5)
            wet: Wet (chorus) level (default 0.5)
            dry: Dry (original) level (default 1.0)
            
        Returns:
            Stereo signal with chorus applied
        """
        n_samples = len(signal)
        t = np.arange(n_samples) / sample_rate
        
        # Initialize output for chorus voices
        chorus_L = np.zeros_like(signal[:, 0], dtype=np.float32)
        chorus_R = np.zeros_like(signal[:, 1], dtype=np.float32)
        
        # Create multiple chorus voices with different LFO phases
        for voice in range(num_voices):
            # Phase offset for each voice to create stereo width
            phase_offset = (2 * np.pi * voice) / num_voices
            
            # LFO modulation for delay time
            lfo = np.sin(2 * np.pi * rate_hz * t + phase_offset)
            mod_delay_ms = delay_ms + depth * lfo
            
            # Convert to samples and ensure it's positive
            mod_delay_samples = np.clip(mod_delay_ms * sample_rate / 1000, 0, None).astype(int)
            
            # Create delayed versions with varying delays
            voice_L = np.zeros_like(signal[:, 0])
            voice_R = np.zeros_like(signal[:, 1])
            
            for i in range(n_samples):
                delay_idx = i - mod_delay_samples[i]
                if delay_idx >= 0:
                    voice_L[i] = signal[delay_idx, 0]
                    voice_R[i] = signal[delay_idx, 1]
            
            # Add to chorus mix with slight panning variations
            pan_L = 0.5 + 0.5 * np.cos(phase_offset)
            pan_R = 0.5 + 0.5 * np.sin(phase_offset)
            
            chorus_L += voice_L * pan_L
            chorus_R += voice_R * pan_R
        
        # Normalize chorus voices
        if num_voices > 0:
            chorus_L /= num_voices
            chorus_R /= num_voices
        
        # Mix dry and wet signals
        out_L = dry * signal[:, 0] + wet * chorus_L
        out_R = dry * signal[:, 1] + wet * chorus_R
        
        # Combine and clip
        stereo_out = np.stack([out_L, out_R], axis=1)
        stereo_out = np.clip(stereo_out, -1.0, 1.0)
        return stereo_out
