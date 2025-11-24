import wave
import numpy as np
import sys

def check_wav(filename):
    try:
        with wave.open(filename, 'r') as wav_file:
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            
            print(f"File: {filename}")
            print(f"Channels: {n_channels}")
            print(f"Sample Width: {sampwidth}")
            print(f"Frames: {n_frames}")
            
            frames = wav_file.readframes(n_frames)
            dtype = np.int16 if sampwidth == 2 else np.int8
            samples = np.frombuffer(frames, dtype=dtype)
            
            max_val = np.max(np.abs(samples))
            print(f"Max Amplitude: {max_val}")
            
            if max_val == 0:
                print("WARNING: File is completely silent!")
            else:
                print("File has audio data.")
                
    except Exception as e:
        print(f"Error reading WAV: {e}")

if __name__ == "__main__":
    check_wav("polyphonic_output.wav")
