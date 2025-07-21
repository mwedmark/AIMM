Projecy currently has the following features:

- Single waveform LFO with waveforms: sine/saw/triangle/square to choose from.
- Pre-rendered Wave files, tha renders to disk and then is played in non-realtime.
- Several different format for music:
  - Direct input using a Python array of tones. I had 3 different hand-coded tracks initially (base/melody/drums) and only a single monophony on each channel. Each tone is strung together in order.
  - Added a format to support chords (same as above but with an array of tone wtihin the array), still non-overlapping start/end of tones on the same channels.
  - Added a third interpreter that takes a MIDI-file and converts it to a normalized format shared for all formats. End-output is the same for all.
  - First 2 formats is easier for hand-written notes (test python note code) and MIDI better for playing lots of already available music or importing played MIDI music.
- Top mix Reverb effect implemented. Parameters can be controlled.
- Volume can be set on each channel.
- MIDI:
  - Listens to note on/note off, Velocity
- MIDI instruments:
    - First an array with settings for: A/D/S/R/Waveform/Reverb was created for each of the 128 MIDI instruments
    - Next some specific instrument were introduced with their own methods that gives more control to which kind of algorithm is best suited for each instrument.
    - Currently the specific instruments are: Grand Piano, Stel Drums and Acoustic Guitar
 
TODO:
  - ** Make it possible to set master Volume for each instrument to be able to mix everything correctly (remove Reverb setting per instrument should be mapped to MIDI channel)
  - ** Map Panning to MIDI signals for each MIDI-channel
  - ** Map Reverb settings to that MIDI effect for ecah channel
  - ** Implement MIDI Drums
  - Add support for Wavetable Synthesis. Either simple format with its own fileformat or direct support for SF2 files.
  - Add more effects, start with Chorus and map that effect against MIDI chorus.
  - Add Delay effect
  - Trouble-shoot the Organ files that still are quiet although they have sound in other MIDI players.
  - Add real-time playing/mixing of WAV files
  - Add mapping to MIDI to be able to play directly with our sounds.
  - Add map to MIDI controllers to be able to play around with sound settings using MIDI panels (ADSR, waveform and more) while playing
  - Add simple Arpeggiator using built-in format (Use Tone-Offset instead of direct tones to be able to control Arrpegio live)
  - Add support for more than one LFO per sound
  - Add support for filters/resonance
  - Add support for FM
  - Add support for Pitchbend
  - Add Support for Vibrato
