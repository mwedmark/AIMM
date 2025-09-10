import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Load the uploaded hi-hat sample again after reset
sample_path = "Closed Hi-hat.wav"
hat_wave, sr = sf.read(sample_path)

# If stereo, take mono mix
if hat_wave.ndim > 1:
    hat_wave = hat_wave.mean(axis=1)

# Normalize
hat_wave = hat_wave / np.max(np.abs(hat_wave))

# FFT for spectral content
fft_size = 8192
freqs = np.fft.rfftfreq(fft_size, 1/sr)
spectrum = np.abs(np.fft.rfft(hat_wave[:fft_size] * np.hanning(len(hat_wave[:fft_size]))))

# Plot waveform and spectrum
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hat_wave[:2000])
plt.title("Hi-hat waveform (first 2000 samples)")

plt.subplot(1,2,2)
plt.semilogx(freqs, 20*np.log10(spectrum+1e-6))
plt.title("Hi-hat spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.tight_layout()
plt.show()

(sr, len(hat_wave))
