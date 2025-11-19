"""
MidiParser - Handles MIDI file parsing and event conversion
"""
from mido import MidiFile
from collections import defaultdict
from instruments import INSTRUMENT_PRESETS


# Global constants (imported from main)
DRUM_MIDI_MAP = {
    35: 'kick', 36: 'kick',
    38: 'snare', 40: 'snare',
    42: 'hat', 44: 'hat', 46: 'hat'
}


class MidiParser:
    """Encapsulates MIDI file parsing and event conversion."""
    
    def __init__(self, beats_per_second):
        """Initialize MidiParser.
        
        Args:
            beats_per_second: Tempo in beats per second
        """
        self.beats_per_second = beats_per_second
        self.channel_volume = defaultdict(lambda: 1.0)
        self.channel_expression = defaultdict(lambda: 1.0)
        self.channel_pan = {}
    
    @staticmethod
    def note_len_to_beats(n):
        """Convert note length to beats.
        
        Args:
            n: Note length (e.g., 4 = quarter note)
            
        Returns:
            Duration in beats
        """
        return 4 / n
    
    @staticmethod
    def get_note_frequency(note):
        """Convert note name or MIDI number to frequency.
        
        Args:
            note: Either MIDI note number (int) or note name (str, e.g., 'C4')
            
        Returns:
            Frequency in Hz
        """
        if isinstance(note, int):  # MIDI note number
            A4_freq = 440.0
            A4_key = 69
            return A4_freq * 2 ** ((note - A4_key) / 12)
        
        if note == 'P':
            return 0.0
        
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
    
    def normalize_events(self, events):
        """Normalize different event formats into a unified list of note events.
        
        Args:
            events: List of events in various formats (tuples or dicts)
            
        Returns:
            List of normalized event dictionaries
        """
        normalized = []
        current_beat = 0.0
        for item in events:
            if isinstance(item, tuple):
                notes, duration = item
                if isinstance(notes, str):
                    notes = [notes]
                duration_beats = self.note_len_to_beats(duration)
                for note in notes:
                    normalized.append({
                        'notes': [note],
                        'start_beats': current_beat,
                        'duration_beats': duration_beats,
                        'channel': -1
                    })
                current_beat += duration_beats
            elif isinstance(item, dict):
                notes = item['notes'] if isinstance(item['notes'], list) else [item['notes']]
                start_beats = sum(self.note_len_to_beats(x) for x in item['start']) if isinstance(item['start'], list) else item['start']
                duration_beats = self.note_len_to_beats(item['length']) if isinstance(item['length'], int) else item['length']
                for note in notes:
                    normalized.append({
                        'notes': [note],
                        'start_beats': start_beats,
                        'duration_beats': duration_beats,
                        'channel': -1
                    })
        return normalized
    
    def midi_extract_drums(self, midi_path, allowed_notes=None):
        """Extract drum events from MIDI file (channel 9).
        
        Args:
            midi_path: Path to MIDI file
            allowed_notes: Optional list of allowed MIDI note numbers
            
        Returns:
            List of drum event dictionaries
        """
        mid = MidiFile(midi_path)
        events = []
        time = 0.0
        active_notes = {}

        for msg in mid:
            time += msg.time
            if msg.type == 'note_on' and msg.channel == 9 and msg.velocity > 0:
                drum_type = DRUM_MIDI_MAP.get(msg.note)
                if drum_type:
                    events.append({
                        'notes': [drum_type],
                        'start_beats': time * self.beats_per_second,
                        'duration_beats': 0.1,  # short fixed duration for drums
                        'channel': 9,
                        'midinote': msg.note
                    })
        return events
    
    def midi_to_events(self, midi_path, allowed_channels=None, default_instrument=0):
        """Parse MIDI file and convert to event list.
        
        Args:
            midi_path: Path to MIDI file
            allowed_channels: Optional list of allowed MIDI channels
            default_instrument: Default instrument program number
            
        Returns:
            List of event dictionaries
        """
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
                if channel_program.get(msg.channel) != msg.program:
                    instrument_name = INSTRUMENT_PRESETS.get(msg.program, ("unknown",))[6]
                    print(f"Channel {msg.channel + 1} selected instrument: {instrument_name}")
                channel_program[msg.channel] = msg.program
            elif msg.type == 'control_change' and msg.control == 7:
                self.channel_volume[msg.channel] = msg.value / 127.0 
            elif msg.type == 'control_change':
                if msg.control == 7:  # Volume MSB
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
                        'program': channel_program.get(msg.channel, default_instrument),
                        'volume': final_volume
                    })
        return events
