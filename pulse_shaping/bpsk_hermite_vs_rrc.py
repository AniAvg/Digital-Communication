import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite

num_bits = 10
sps = 8
beta = 0.35
span = 6
Ts = 1
snr_db = 5

bits = np.random.randint(0, 2, num_bits)
symbols = 2 * bits - 1

upsampled = np.zeros(len(symbols) * sps)
upsampled[::sps] = symbols

time = np.arange(-span / 2, span / 2 + 1 / sps, 1 / sps)

def root_raised_cosine(t, Ts, beta):
    rrc = np.zeros_like(t)
    eps = 1e-8

    for i in range(len(t)):
        ti = t[i]
        if abs(ti) < eps:
            rrc[i] = (1 / Ts) * (1 + beta * (4 / np.pi - 1))
        elif abs(abs(ti) - Ts / (4 * beta)) < eps:
            rrc[i] = (beta / (Ts * np.sqrt(2))) * (
                    (1 + 2/np.pi) * np.sin(np.pi / (4 * beta)) +
                    (1 - 2/np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            numerator = (
                np.sin(np.pi * ti * (1 - beta) / Ts) +
                4 * beta * ti / Ts *
                np.cos(np.pi * ti * (1 + beta) / Ts))
            denominator = (
                np.pi * ti / Ts *
                (1 - (4 * beta * ti / Ts) ** 2))
            rrc[i] = (1/Ts) * numerator / denominator
    return rrc

def hermite_pulse(t, order=0, Ts=1):
    t_scaled = t / Ts
    Hn = hermite(order)
    pulse = Hn(t_scaled) * np.exp(-t_scaled ** 2 / 2)
    pulse /= np.sqrt(np.sum(pulse ** 2))
    return pulse

def add_awgn(signal, snr_db):
    snr_lin = 10 ** (snr_db / 10)
    power = np.mean(signal ** 2)
    noise_power = power / snr_lin
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

rrc_filter = root_raised_cosine(time, Ts, beta)
herm_filter = hermite_pulse(time, order=0, Ts=Ts)

tx_rrc = np.convolve(upsampled, rrc_filter)
tx_herm = np.convolve(upsampled, herm_filter)

tx_rrc_noisy = add_awgn(tx_rrc, snr_db)
tx_herm_noisy = add_awgn(tx_herm, snr_db)

y_rrc = np.convolve(tx_rrc_noisy, rrc_filter[::-1])
y_herm = np.convolve(tx_herm_noisy, herm_filter[::-1])

delay_rrc = (len(rrc_filter) - 1) // 2
delay_herm = (len(herm_filter) - 1) // 2
total_delay_rrc = 2 * delay_rrc
total_delay_herm = 2 * delay_herm

sample_indices_rrc = total_delay_rrc + np.arange(len(symbols)) * sps
sample_indices_herm = total_delay_herm + np.arange(len(symbols)) * sps

samples_rrc = y_rrc[sample_indices_rrc]
samples_herm = y_herm[sample_indices_herm]

detected_rrc = np.sign(samples_rrc)
detected_herm = np.sign(samples_herm)

print("Original Symbols: ", symbols)
print("RRC Detected:     ", detected_rrc)
print("Hermite Detected: ", detected_herm)

plt.figure(figsize=(12, 4))
plt.plot(time, rrc_filter, label="RRC Pulse")
plt.plot(time, herm_filter, label="Hermite Pulse")
plt.title("Pulse Shapes")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.plot(np.arange(len(tx_rrc))/sps, tx_rrc, label="RRC Tx")
plt.plot(np.arange(len(tx_herm))/sps, tx_herm, label="Hermite Tx")
plt.title("Transmit Signals (No Noise)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(np.arange(len(tx_rrc_noisy))/sps, tx_rrc_noisy, label="RRC Tx + Noise")
plt.plot(np.arange(len(tx_herm_noisy))/sps, tx_herm_noisy, label="Hermite Tx + Noise")
plt.title("Transmit Signals with Noise")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12,4))
plt.plot(np.arange(len(y_rrc))/sps, y_rrc, label="RRC MF Output")
plt.plot(np.arange(len(y_herm))/sps, y_herm, label="Hermite MF Output")
plt.stem(sample_indices_rrc/sps, samples_rrc, linefmt='r-', markerfmt='ro', basefmt=' ', label="RRC Samples")
plt.stem(sample_indices_herm/sps, samples_herm, linefmt='g-', markerfmt='go', basefmt=' ', label="Hermite Samples")
plt.title("Matched Filter Output and Samples")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()
