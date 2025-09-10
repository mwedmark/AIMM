import os
import sys
import wave
import numpy as np
from collections import defaultdict
from MIDIProcessor import MIDIProcessor
from Synthesizer import Synthesizer
from AudioMixer import AudioMixer
from mido import MidiFile
import MidiInstrumentsSpec  # Import the module
import random

def main():
    if len(sys.argv) < 2:
        midi_path = "MIDIs/UnderTheSea.mid"
    else:
        midi_path = sys.argv[1]

    if not os.path.exists(midi_path):
        print(f"File not found: {midi_path}")
        sys.exit(1)

    print(f"Processing MIDI: {midi_path}")

    # Initialize classes
    use_sample_drums = False  # Change to True to use sample-based drums
    
    sample_rate = 44100
    tempo_bpm = 110
    midi_processor = MIDIProcessor(sample_rate, tempo_bpm)
    synthesizer = Synthesizer(sample_rate, use_sample_drums)
    audio_mixer = AudioMixer(sample_rate)

    # Process MIDI
    midi_events = midi_processor.extract_events(midi_path, allowed_channels=list(range(16)))
    drum_events = midi_processor.extract_drum_events(midi_path)

    # Generate note frequencies
    note_freqs = {}
    for event in midi_events + drum_events:
        for note in event['notes']:
            if note not in note_freqs:
                note_freqs[note] = midi_processor.get_note_frequency(note)

    # Synthesize melody and drums
    melody_events_by_channel = defaultdict(list)
    for event in midi_events:
        melody_events_by_channel[event['channel']].append(event)

    for channel, events in melody_events_by_channel.items():
        track, pan = synthesizer.synthesize_track(events, note_freqs, midi_processor.seconds_per_beat, MidiInstrumentsSpec)
        audio_mixer.add_track(track, volume=0.3, pan=pan, apply_reverb=True)

    drum_track = synthesizer.synthesize_track(drum_events, note_freqs, midi_processor.seconds_per_beat, MidiInstrumentsSpec)[0]
    audio_mixer.add_track(drum_track, volume=0.005, pan=0.5, apply_reverb=True)

    # Mix and render to WAV
    output_path = "output.wav"
    audio_mixer.render_to_wav(output_path)

    # Play the output
    print(f"Playing: {output_path}")
    import winsound
    winsound.PlaySound(output_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    input("Press ENTER to stop playback...\n")
    winsound.PlaySound(None, winsound.SND_PURGE)

if __name__ == "__main__":
    main()