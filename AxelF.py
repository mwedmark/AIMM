import numpy as np
import wave
import winsound
import sys
import os

# Import the new classes
from audio_processor import AudioProcessor
from midi_parser import MidiParser
from synthesizer import Synthesizer

# ================== Settings ==================
sample_rate = 44100
TEMPO_BPM = 110
BEATS_PER_SECOND = TEMPO_BPM / 60

DEFAULT_INSTRUMENT = 0  # fallback if no program_change
USE_SAMPLED_DRUMS = False  # switch here

# ================== Example Tracks ==================
melody_track = [
    ('C3', 4), ('D#3', 6), ('C3', 8), ('C3', 16), ('F3', 8), ('C3', 8), ('A#2', 8), ('C3', 4),
    ('G3', 6), ('C3', 8), ('C3', 16), ('G#3', 8), ('G3', 8), ('D#3', 8), ('C3', 8), ('G3', 8),
    ('C4', 8), ('C3', 16), ('A#2', 8), ('A#2', 16), ('G2', 8), ('D3', 8), ('C3', 2)
]

bass_track = [
    ('C0', 4), ('C1', 6), ('A#0', 8), ('A#1', 16), ('G0', 8), ('G1', 8), ('A#0', 8),
    ('C0', 4), ('C1', 6), ('P', 8), ('G0', 16), ('G1', 8), ('A#1', 8), ('C1', 8),
    ('G#0', 4), ('G#1', 6), ('A#0', 8), ('A#1', 16), ('G0', 8), ('A#0', 8), ('C0', 8),
    ('C1', 4), ('P', 4), ('P', 16), ('A#1', 16), ('G0', 8), ('F0', 8), ('D#0', 8)
]




def main():
    if len(sys.argv) < 2:
        midi_path = "midis/UnderTheSea.mid"
    else:
        midi_path = sys.argv[1]
        
    if not os.path.exists(midi_path):
        print(f"File not found: {midi_path}")
        sys.exit(1)

    print(f"Playing MIDI: {midi_path}")

    # Initialize the new classes
    midi_parser = MidiParser(BEATS_PER_SECOND)
    synthesizer = Synthesizer(sample_rate, DEFAULT_INSTRUMENT, USE_SAMPLED_DRUMS)
    audio_processor = AudioProcessor()

    # Parse MIDI events
    midi_events = midi_parser.midi_to_events(
        midi_path, 
        allowed_channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        default_instrument=DEFAULT_INSTRUMENT
    )
    normalized_melody = midi_events

    # Build note frequency map
    note_freqs = {}
    all_events = normalized_melody
    for event in all_events:
        for note in event['notes']:
            if note not in note_freqs:
                note_freqs[note] = MidiParser.get_note_frequency(note)
                
    # Synthesize melody channels
    melody_events_by_channel = Synthesizer.group_events_by_channel(normalized_melody)
    melody_stereo = synthesizer.synthesize_channels(
        melody_events_by_channel, note_freqs, 1 / BEATS_PER_SECOND
    )

    # Extract and synthesize drums
    drum_track = midi_parser.midi_extract_drums(midi_path)
    drum_stereo = synthesizer.generate_drum_track(drum_track, 1 / BEATS_PER_SECOND)

    # Mix stereo
    stereo = synthesizer.mix_voices(
        melody_stereo, drum_stereo,
        melody_volume=0.3, melody_reverb=True,
        drums_volume=0.05, drums_reverb=True
    )

    # ================== Final WAV Output ==================
    # Normalize
    max_val = np.max(np.abs(stereo))
    if max_val > 1.0:
        stereo = stereo / max_val

    # Apply soft limiter and normalize RMS
    stereo = audio_processor.soft_limiter_agg(stereo)
    stereo = audio_processor.normalize_rms(stereo)

    stereo = np.clip(stereo, -1.0, 1.0)

    # Save to WAV
    filename = "polyphonic_output.wav"
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        int_samples = (stereo * 32767).astype(np.int16)
        wav_file.writeframes(int_samples.tobytes())

    # Play
    winsound.PlaySound(filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
    input("Press ENTER to stop playback...\n")
    winsound.PlaySound(None, winsound.SND_PURGE)
        

if __name__ == "__main__":
    main()

