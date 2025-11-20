"""
DrumSynthesizer - Handles all drum synthesis (sampled and synthesized)
"""
import numpy as np
from MidiInstrumentsSpec import MidiInstruments


class DrumSynthesizer:
    """Encapsulates drum synthesis logic for both sampled and synthesized drums."""
    
    def __init__(self, sample_rate, use_sampled_drums=False):
        """Initialize DrumSynthesizer.
        
        Args:
            sample_rate: Audio sample rate
            use_sampled_drums: Whether to use sampled drums instead of synthesized
        """
        self.sample_rate = sample_rate
        self.use_sampled_drums = use_sampled_drums
        self.drumkit = None
        
        if self.use_sampled_drums:
            from sampled_drums import SampledDrums
            self.drumkit = SampledDrums(sample_dir="samples", sample_rate=self.sample_rate)
    
    def generate_drum_track(self, normalized_events, seconds_per_beat, drum_pan_map=None):
        """Generate stereo drum track from drum events.
        
        Args:
            normalized_events: List of drum event dictionaries
            seconds_per_beat: Seconds per beat (tempo)
            drum_pan_map: Optional dictionary for drum panning
            
        Returns:
            Stereo drum track array (N x 2)
        """
        if not normalized_events:
            return np.zeros((1, 2), dtype=np.float32)

        end_time = max(e['start_beats'] + e['duration_beats'] for e in normalized_events)
        total_samples = int(end_time * seconds_per_beat * self.sample_rate)
        stereo_track = np.zeros((total_samples, 2), dtype=np.float32)

        if drum_pan_map is None:
            drum_pan_map = {}

        for e in normalized_events:
            drum_type = e['notes'][0]
            start_sample = int(e['start_beats'] * seconds_per_beat * self.sample_rate)
            duration = e['duration_beats'] * seconds_per_beat

            key = (drum_type, e.get('channel', 9))
            midinote = e.get('midinote', 9)
            pan = drum_pan_map.get(key, np.random.uniform(0.3, 0.7))
            drum_pan_map[key] = pan
            
            velocity = e.get("velocity", 100)

            if self.use_sampled_drums:
                wave = self.drumkit.trigger(midinote, velocity, pan)
            else:
                wave = MidiInstruments.synth_drum_stereo(drum_type, duration, pan, velocity, self.sample_rate)
                
            end_sample = start_sample + wave.shape[0]

            if end_sample > total_samples:
                wave = wave[:total_samples - start_sample]

            stereo_track[start_sample:start_sample + wave.shape[0]] += wave

        stereo_track = np.clip(stereo_track, -1.0, 1.0)
        return stereo_track
