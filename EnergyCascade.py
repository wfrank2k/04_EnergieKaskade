import numpy as np
import matplotlib.pyplot as plt

# Read time series from input file
with open('input_file.txt', 'r') as f:
    data = f.readlines()

time_series = [float(d.strip()) for d in data]

# Compute the power spectrum using FFT
n = len(time_series)
power_spectrum = np.abs(np.fft.fft(time_series))**2 / n
freq = np.fft.fftfreq(n)

# Calculate wave lengths
wave_lengths = 2 * np.pi / np.abs(freq)

# Compute turbulent energy cascade for each wave length
turbulent_energy = np.zeros(len(wave_lengths))
for i in range(len(wave_lengths)):
    w_min = wave_lengths[i] / 2
    w_max = wave_lengths[i]
    k = 2 * np.pi / wave_lengths[i]
    turbulent_energy[i] = np.sum(power_spectrum[(freq >= k/w_max) & (freq < k/w_min)]) * k

# Plot turbulent energy cascade for each wave length
plt.plot(wave_lengths, turbulent_energy)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Wave Length')
plt.ylabel('Turbulent Energy')
plt.title('Turbulent Energy Cascade')
plt.show()
