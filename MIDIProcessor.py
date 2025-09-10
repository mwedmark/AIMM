from builtins import isinstance, int, ValueError, round, hasattr
from collections import defaultdict
import random
from mido import MidiFile

class MIDIProcessor:
    def __init__(self, sample_rate=44100, tempo_bpm=110):
        self.sample_rate = sample_rate
        self.tempo_bpm = tempo_bpm
        self.beats_per_second = tempo_bpm / 60
        self.seconds_per_beat = 1 / self.beats_per_second
        self.channel_volume = defaultdict(lambda: 1.0)
        self.channel_expression = defaultdict(lambda: 1.0)
        self.channel_pan = {}
        self.default_instrument = 0
        self.drum_midi_map = {
            35: 'kick', 36: 'kick',
            38: 'snare', 40: 'snare',
            42: 'hat', 44: 'hat', 46: 'hat'
        }
    
    def get_channel_pan(self, channel):
        if channel not in self.channel_pan:
            self.channel_pan[channel] = round(random.uniform(0.3, 0.7), 2)
        return self.channel_pan[channel]
        
    def get_note_frequency(self, note):
            # Handle MIDI note numbers
            if isinstance(note, int):  # MIDI note number
                A4_freq = 440.0
                A4_key = 69
                return A4_freq * 2 ** ((note - A4_key) / 12)
        
            # Handle special cases
            if note == 'P':
                return 0.0
            
            # Handle drum sounds with fixed frequencies
            drum_frequencies = {
                'kick': 60.0,
                'snare': 150.0,
                'hat': 300.0
            }
            if note in drum_frequencies:
                return drum_frequencies[note]
        
            # Normal note processing for notes in format like 'C4', 'F#3', etc.
            try:
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
            except (ValueError, IndexError):
                raise ValueError(f"Unrecognized note format: {note}")
    
    def extract_events(self, midi_path, allowed_channels=None):
        mid = MidiFile(midi_path)
        events = []
        time = 0.0
        active_notes = {}
        channel_program = {}
        
        for msg in mid:
            time += msg.time
            if msg.type == 'note_on' and msg.channel == 9 and msg.velocity > 0:
                continue
            if not msg.is_meta and hasattr(msg, 'channel'):
                if allowed_channels is not None and msg.channel not in allowed_channels:
                    continue
            
            if msg.type == 'program_change':
                channel_program[msg.channel] = msg.program
            elif msg.type == 'control_change':
                if msg.control == 7:  # Volume
                    self.channel_volume[msg.channel] = msg.value / 127.0
                elif msg.control == 10:  # PAN
                    self.channel_pan[msg.channel] = msg.value / 127.0
            elif msg.type == 'note_on' and msg.velocity > 0:
                active_notes[(msg.note, msg.channel)] = time
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in active_notes:
                    start = active_notes.pop(key)
                    duration = time - start
                    vol = self.channel_volume.get(msg.channel, 1.0)
                    vol = max(vol, 0.05)
                    velocity = msg.velocity / 127.0
                    if velocity == 0 and msg.type == 'note_on':
                        velocity = 100
                    exp = self.channel_expression.get(msg.channel, 1.0)
                    final_volume = vol * exp * velocity
                    events.append({
                        'notes': [msg.note],
                        'start_beats': start * self.beats_per_second,
                        'duration_beats': duration * self.beats_per_second,
                        'channel': msg.channel,
                        'program': channel_program.get(msg.channel, self.default_instrument),
                        'volume': final_volume
                    })
        return events
    
    def extract_drum_events(self, midi_path):
        mid = MidiFile(midi_path)
        events = []
        time = 0.0
        
        for msg in mid:
            time += msg.time
            if msg.type == 'note_on' and msg.channel == 9 and msg.velocity > 0:
                drum_type = self.drum_midi_map.get(msg.note)
                if drum_type:
                    events.append({
                        'notes': [drum_type],
                        'start_beats': time * self.beats_per_second,
                        'duration_beats': 0.1,  # short fixed duration for drums
                        'channel': 9,
                        'midinote': msg.note
                    })
        return events