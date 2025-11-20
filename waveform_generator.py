"""
WaveformGenerator - Handles all waveform generation for instruments
"""
import numpy as np
from audio_processor import AudioProcessor
from MidiInstrumentsSpec import MidiInstruments


class WaveformGenerator:
    """Encapsulates all waveform generation logic including basic waveforms
    and specialized instrument sounds."""
    
    def __init__(self, sample_rate):
        """Initialize WaveformGenerator.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor()
    
    def generate_wave(self, freq, duration, wave_type='square', 
                     a_sec=0.02, d_sec=0.05, s_level=0.4, r_sec=0.08):
        """Generate a basic waveform with ADSR envelope.
        
        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            wave_type: Type of waveform ('sine', 'square', 'triangle', 'saw')
            a_sec: Attack time in seconds
            d_sec: Decay time in seconds
            s_level: Sustain level (0.0 to 1.0)
            r_sec: Release time in seconds
            
        Returns:
            Generated waveform array
        """
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
        envelope = self.audio_processor.adsr_envelope(samples, self.sample_rate, a_sec, d_sec, s_level, r_sec)
        shaped = wave * envelope

        # Optional lowpass to reduce aliasing (especially on saw/square)
        if wave_type in ['saw', 'square']:
            shaped = self.audio_processor.lowpass_filter(shaped, cutoff=8000, sr=self.sample_rate)

        return shaped.astype(np.float32)
    
    def generate_piano(self, freq, duration):
        """Generate piano waveform.
        
        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            
        Returns:
            Piano waveform array
        """
        return MidiInstruments.generate_wave_piano(freq, duration, self.sample_rate)
    
    def generate_nylon_guitar(self, freq, duration):
        """Generate nylon guitar waveform.
        
        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            
        Returns:
            Nylon guitar waveform array
        """
        return MidiInstruments.generate_wave_nylon_guitar(freq, duration, self.sample_rate)
    
    def generate_steel_drum(self, freq, duration):
        """Generate steel drum waveform.
        
        Args:
            freq: Frequency in Hz
            duration: Duration in seconds
            
        Returns:
            Steel drum waveform array
        """
        return MidiInstruments.generate_wave_steeldrum(freq, duration, self.sample_rate)
    
    def generate_for_program(self, program, freq, duration, waveform, a_sec, d_sec, s_level, r_sec):
        """Generate waveform for a specific MIDI program number.
        
        Args:
            program: MIDI program number
            freq: Frequency in Hz
            duration: Duration in seconds
            waveform: Basic waveform type for non-specialized instruments
            a_sec: Attack time in seconds
            d_sec: Decay time in seconds
            s_level: Sustain level (0.0 to 1.0)
            r_sec: Release time in seconds
            
        Returns:
            Generated waveform array
        """
        if program == 0:  # Acoustic Grand Piano
            return self.generate_piano(freq, duration)
        elif program == 24:  # Acoustic Guitar (nylon)
            return self.generate_nylon_guitar(freq, duration)
        elif program == 114:  # Steel Drums
            return self.generate_steel_drum(freq, duration)
        else:
            return self.generate_wave(freq, duration, waveform, a_sec, d_sec, s_level, r_sec)
