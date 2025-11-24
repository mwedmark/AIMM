import mido
from mido import Message, MidiFile, MidiTrack
import os

def create_demo():
    # Ensure midis directory exists
    if not os.path.exists('midis'):
        os.makedirs('midis')

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # We will use different channels to demonstrate different effect settings
    # because the current synthesizer applies effects per-channel.
    #
    # Sequence:
    # 1. Channel 0: Dry (No effects)
    # 2. Channel 1: Light Reverb
    # 3. Channel 2: Heavy Reverb
    # 4. Channel 3: Light Chorus
    # 5. Channel 4: Heavy Chorus
    # 6. Channel 5: Reverb + Chorus

    demos = [
        {'ch': 0, 'rev': 0, 'cho': 0, 'name': 'Dry'},
        {'ch': 1, 'rev': 40, 'cho': 0, 'name': 'Light Reverb'},
        {'ch': 2, 'rev': 110, 'cho': 0, 'name': 'Heavy Reverb'},
        {'ch': 3, 'rev': 0, 'cho': 50, 'name': 'Light Chorus'},
        {'ch': 4, 'rev': 0, 'cho': 110, 'name': 'Heavy Chorus'},
        {'ch': 5, 'rev': 70, 'cho': 70, 'name': 'Reverb + Chorus'},
    ]
    
    # Initial Setup (Time = 0)
    # Set Instrument (Program 5: Electric Piano 1) for all channels
    # And set initial effect levels
    for d in demos:
        ch = d['ch']
        # Program 5 is Electric Piano 1 (0-indexed in code usually, but let's use 4 for EP1 or 0 for Grand Piano)
        # Let's use 5 (Electric Piano 2) for a nice synth sound
        track.append(Message('program_change', program=5, channel=ch, time=0))
        
        # Set Effects
        track.append(Message('control_change', control=91, value=d['rev'], channel=ch, time=0))
        track.append(Message('control_change', control=93, value=d['cho'], channel=ch, time=0))
        
        # Set Volume and Pan
        track.append(Message('control_change', control=7, value=100, channel=ch, time=0))
        track.append(Message('control_change', control=10, value=64, channel=ch, time=0)) # Center pan

    # Melody notes (C Major Arpeggio: C4, E4, G4, C5)
    notes = [60, 64, 67, 72]
    note_duration = 240  # Eighth note (assuming 480 ticks/beat)
    rest_duration = 480  # Quarter note rest
    
    # Generate Sequence
    # We rely on delta times. The first event of a block needs to account for the time passed.
    
    last_time = 0
    
    for i, d in enumerate(demos):
        ch = d['ch']
        print(f"Generating section: {d['name']} (Ch {ch})")
        
        # Arpeggio
        for j, note in enumerate(notes):
            # Note On
            # First note of the block has a delay if it's not the very first block
            dt = rest_duration if (j == 0 and i > 0) else 0
            track.append(Message('note_on', note=note, velocity=100, channel=ch, time=dt))
            
            # Note Off
            track.append(Message('note_off', note=note, velocity=0, channel=ch, time=note_duration))
            
        # Chord
        # On (all at once)
        track.append(Message('note_on', note=60, velocity=90, channel=ch, time=0))
        track.append(Message('note_on', note=64, velocity=90, channel=ch, time=0))
        track.append(Message('note_on', note=67, velocity=90, channel=ch, time=0))
        track.append(Message('note_on', note=72, velocity=90, channel=ch, time=0))
        
        # Off (after hold)
        hold_duration = note_duration * 4
        track.append(Message('note_off', note=60, velocity=0, channel=ch, time=hold_duration))
        track.append(Message('note_off', note=64, velocity=0, channel=ch, time=0))
        track.append(Message('note_off', note=67, velocity=0, channel=ch, time=0))
        track.append(Message('note_off', note=72, velocity=0, channel=ch, time=0))

    filename = 'midis/effects_demo.mid'
    mid.save(filename)
    print(f"Successfully created {filename}")

if __name__ == '__main__':
    create_demo()
