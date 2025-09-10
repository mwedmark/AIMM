from mido import Message, MidiFile, MidiTrack, MetaMessage

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(MetaMessage('set_tempo', tempo=500000))  # 120 BPM
track.append(MetaMessage('time_signature', numerator=4, denominator=4))

# Two pianos: one far right, one far left
track.append(Message('program_change', program=0, channel=0))
track.append(Message('program_change', program=0, channel=1))
track.append(Message('control_change', control=10, value=127, channel=0))  # Far right
track.append(Message('control_change', control=10, value=0, channel=1))    # Far left

# Right channel note
track.append(Message('note_on', note=60, velocity=64, channel=0, time=0))
track.append(Message('note_off', note=60, velocity=64, channel=0, time=480))

# Left channel note
track.append(Message('note_on', note=64, velocity=64, channel=1, time=0))
track.append(Message('note_off', note=64, velocity=64, channel=1, time=480))

mid.save("piano_left_right.mid")
