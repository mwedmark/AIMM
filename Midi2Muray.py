from mido import MidiFile
import math

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_number_to_name(note_number):
    note = NOTE_NAMES[note_number % 12]
    octave = note_number // 12 - 1
    return f"{note}{octave}"

def extract_melody_track(mid):
    for track in mid.tracks:
        if any(msg.type == 'note_on' for msg in track):
            return track
    return None

def quantize_duration(duration_ticks, ticks_per_beat):
    # Common durations in order (whole to 1/64 note)
    durations = [
        (1, ticks_per_beat * 4),
        (2, ticks_per_beat * 2),
        (4, ticks_per_beat),
        (8, ticks_per_beat // 2),
        (16, ticks_per_beat // 4),
        (32, ticks_per_beat // 8),
        (64, ticks_per_beat // 16),
    ]

    # Find closest duration
    closest = min(durations, key=lambda x: abs(x[1] - duration_ticks))
    return closest[0]


def parse_melody(track, ticks_per_beat, channel_filter=0):
    melody = []
    ongoing_notes = {}

    current_time = 0
    for msg in track:
        current_time += msg.time
        if msg.type in ('note_on', 'note_off'):
            if channel_filter is not None and msg.channel != channel_filter:
                continue  # Skip if not the selected channel

            if msg.type == 'note_on' and msg.velocity > 0:
                ongoing_notes[msg.note] = current_time
            elif msg.note in ongoing_notes:
                start_time = ongoing_notes.pop(msg.note)
                duration_ticks = current_time - start_time
                duration_symbol = quantize_duration(duration_ticks, ticks_per_beat)
                note_name = note_number_to_name(msg.note)
                melody.append((note_name, duration_symbol))
    return melody


def midi_to_melody_array(filename):
    mid = MidiFile(filename)
    ticks_per_beat = mid.ticks_per_beat
    melody_track = extract_melody_track(mid)
    if melody_track is None:
        print("No melody track with note events found.")
        return []

    melody = parse_melody(melody_track, ticks_per_beat)
    return melody

def save_to_file(melody, filename="melody_output.py", var_name="melody_track"):
    with open(filename, 'w') as f:
        f.write(f"{var_name} = [\n    ")
        for i, (note, duration) in enumerate(melody):
            f.write(f"('{note}', {duration}), ")
            if (i + 1) % 8 == 0 and i != len(melody) - 1:
                f.write("\n    ")
        f.write("\n]\n")
    print(f"Saved to {filename}")


# Example usage:
if __name__ == "__main__":
    filename = "test_scale.mid"
    melody_track = midi_to_melody_array(filename)
    save_to_file(melody_track)
