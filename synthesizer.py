"""
Synthesizer - Manages waveform generation and track synthesis
"""
import numpy as np
import random
from collections import defaultdict
from instruments import INSTRUMENT_PRESETS
from audio_processor import AudioProcessor
from waveform_generator import WaveformGenerator
from drum_synthesizer import DrumSynthesizer


class Synthesizer:
    """Encapsulates waveform generation and audio synthesis."""
    
    def __init__(self, sample_rate, default_instrument=0, use_sampled_drums=False):
        """Initialize Synthesizer.
        
        Args:
            sample_rate: Audio sample rate
            default_instrument: Default MIDI instrument program number
            use_sampled_drums: Whether to use sampled drums instead of synthesized
        """
        self.sample_rate = sample_rate
        self.default_instrument = default_instrument
        self.use_sampled_drums = use_sampled_drums
        self.channel_pan = {}
        self.audio_processor = AudioProcessor()
        self.waveform_generator = WaveformGenerator(sample_rate)
        self.drum_synthesizer = DrumSynthesizer(sample_rate, use_sampled_drums)
    
    def get_channel_pan(self, channel):
        """Get or assign pan value for a channel.
        
        Args:
            channel: MIDI channel number
            
        Returns:
            Pan value (0.0 = left, 1.0 = right)
        """
        if channel not in self.channel_pan:
            self.channel_pan[channel] = round(random.uniform(0.3, 0.7), 2)
        return self.channel_pan[channel]
    

    
    @staticmethod
    def group_events_by_channel(events):
        """Group events by their MIDI channel.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Dictionary mapping channel numbers to event lists
        """
        grouped = defaultdict(list)
        for event in events:
            channel = event.get('channel', 0)
            grouped[channel].append(event)
        return grouped
    
    def synthesize_track(self, events, note_freqs, seconds_per_beat):
        """Synthesize a single track from events.
        
        Args:
            events: List of event dictionaries
            note_freqs: Dictionary mapping notes to frequencies
            seconds_per_beat: Seconds per beat (tempo)
            
        Returns:
            Tuple of (audio track array, average pan value)
        """
        if not events:
            return np.zeros(1, dtype=np.float32), self.get_channel_pan(0)

        end_time = max(e['start_beats'] + e['duration_beats'] for e in events)
        total_samples = int(end_time * seconds_per_beat * self.sample_rate)
        track = np.zeros(total_samples, dtype=np.float32)

        used_channels = set()

        for event in events:
            pan = event.get('pan')
            channel = event.get('channel', 0)
            if pan is not None:
                self.channel_pan[channel] = pan
            program = event.get('program', self.default_instrument)
            used_channels.add(channel)

            waveform, a, d, s, r, mix_vol, name = INSTRUMENT_PRESETS.get(
                program, INSTRUMENT_PRESETS[self.default_instrument]
            )
            
            for note in event['notes']:
                from midi_parser import MidiParser
                freq = note_freqs.get(note, MidiParser.get_note_frequency(note))
                if freq == 0.0:
                    continue

                start_sample = int(event['start_beats'] * seconds_per_beat * self.sample_rate)
                duration = event['duration_beats'] * seconds_per_beat
                
                wave = self.waveform_generator.generate_for_program(
                    program, freq, duration, waveform, a, d, s, r
                )
                
                wave = self.audio_processor.apply_fade_out(wave, self.sample_rate)

                end_sample = start_sample + len(wave)
                if end_sample > len(track):
                    wave = wave[:len(track) - start_sample]

                volume = event.get('volume', 1.0)
                volume = max(0.0, min(1.0, volume))
                voice_gain = 0.2
                track[start_sample:start_sample + len(wave)] += wave * volume * voice_gain * mix_vol

        # If multiple channels used, average their pans
        avg_pan = sum(self.get_channel_pan(ch) for ch in used_channels) / len(used_channels)
        return track, avg_pan
    
    def synthesize_channels(self, events_by_channel, note_freqs, seconds_per_beat):
        """Synthesize and pan multiple channels into stereo mix.
        
        Args:
            events_by_channel: Dictionary mapping channels to event lists
            note_freqs: Dictionary mapping notes to frequencies
            seconds_per_beat: Seconds per beat (tempo)
            
        Returns:
            Stereo audio array (N x 2)
        """
        channel_tracks = []
        volume_scale = 1.0 / max(4, len(events_by_channel))

        for ch, ch_events in events_by_channel.items():
            audio, _ = self.synthesize_track(ch_events, note_freqs, seconds_per_beat)
            audio *= volume_scale
            pan = self.get_channel_pan(ch)
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
    
    def generate_drum_track(self, normalized_events, seconds_per_beat, drum_pan_map=None):
        """Generate stereo drum track from drum events.
        
        Args:
            normalized_events: List of drum event dictionaries
            seconds_per_beat: Seconds per beat (tempo)
            drum_pan_map: Optional dictionary for drum panning
            
        Returns:
            Stereo drum track array (N x 2)
        """
        return self.drum_synthesizer.generate_drum_track(
            normalized_events, seconds_per_beat, drum_pan_map
        )
    
    def mix_voices(self, melody_stereo, drums_stereo, 
                   melody_volume=1.0, melody_reverb=True,
                   drums_volume=1.0, drums_reverb=False):
        """Mix melody and drum tracks with optional reverb.
        
        Args:
            melody_stereo: Stereo melody track
            drums_stereo: Stereo drums track
            melody_volume: Volume scaling for melody
            melody_reverb: Whether to apply reverb to melody
            drums_volume: Volume scaling for drums
            drums_reverb: Whether to apply reverb to drums
            
        Returns:
            Mixed stereo audio
        """
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
            m = self.audio_processor.reverb_stereo(m, self.sample_rate)
        if drums_reverb:
            d = self.audio_processor.reverb_stereo(d, self.sample_rate, wet=0.15, lowpass_fc=3000)

        mix = m + d
        mix = np.clip(mix, -1.0, 1.0)
        return mix.astype(np.float32)
