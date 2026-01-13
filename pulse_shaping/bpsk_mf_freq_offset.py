import numpy as np
import matplotlib.pyplot as plt

num_bits = 10
sps = 8
beta = 0.35
span = 6
Ts = 1
freq_offset = 0.05

bits = np.random.randint(0, 2, num_bits)
symbols = 2 * bits - 1

time = np.arange(-span/2, span/2 + 1/sps, 1/sps)

unsampled = np.zeros(len(symbols) * sps)
unsampled[::sps] = symbols


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

def add_frequency_offset(signal, freq_offset, sps):
    n = np.arange(len(signal))
    return signal * np.exp(1j * 2 * np.pi * freq_offset * n / sps)

def freq_offset_correction(signal, freq_est, sps):
    n = np.arange(len(signal))
    return signal * np.exp(-1j * 2 * np.pi * freq_est * n / sps)

def add_awgn(signal, snr_db):
    snr_lin = 10 ** (snr_db / 10)
    power = np.mean(signal ** 2)
    noise_power = power / snr_lin
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise


rrc_filter = root_raised_cosine(time, Ts, beta)
rrc_filter = rrc_filter / np.sqrt(np.sum(rrc_filter ** 2))

tx = np.convolve(unsampled, rrc_filter)
tx = tx.astype(complex)

# adding freq offset
tx_freq_offset = add_frequency_offset(tx, freq_offset, sps)

tx_with_noise = add_awgn(tx, 5)

matched_filter = rrc_filter[::-1]
y = np.convolve(tx_with_noise, matched_filter)


phase_diff = np.angle(y[1:] * np.conj(y[:-1]))
freq_est = np.mean(phase_diff) * sps / (2 * np.pi)
y_corrected = freq_offset_correction(y, freq_est, sps)

delay = (len(rrc_filter) - 1) // 2
total_delay = 2 * delay

sample_indices = total_delay + np.arange(len(symbols)) * sps
samples_before = y[sample_indices]
samples_after = y_corrected[sample_indices]


print("Symbols: ", symbols)
print("Samples before: ", samples_before)
print("Samples after: ", samples_after)

print("True frequency offset:", freq_offset)
print("Estimated frequency offset:", freq_est)
print("Detected symbols after correction:", np.sign(np.real(samples_after)))


def plot_eye(signal, sps, title="Eye Diagram"):
    eye_len = 2 * sps

    for i in range(40):
        start = i * sps
        if start + eye_len < len(signal):
            plt.plot(np.real(signal[start:start+eye_len]), 'b', alpha=0.3)
    plt.title(title)
    plt.xlabel("Samples")
    plt.grid(True)



plt.figure(figsize = (10, 4))
plt.plot(time, rrc_filter)
plt.title("RRC Pulse")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

tx_time = np.arange(len(tx)) / sps
tx_time_noisy = np.arange(len(tx_with_noise)) / sps

plt.figure(figsize = (10, 4))
plt.plot(tx_time, tx)
plt.title("Transmit Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.figure(figsize=(10, 4))
plt.plot(np.real(tx), label="Tx (No Offset)")
plt.plot(np.real(tx_freq_offset), label="Tx (With Freq Offset)", alpha=0.7)
plt.title("Transmit Signal (Time Domain)")
plt.legend()
plt.grid(True)

# Constellation BEFORE correction
plt.figure(figsize = (10, 4))
plt.subplot(1, 2, 1)
plt.plot(np.real(samples_before), np.imag(samples_before), 'o')
plt.title("Constellation BEFORE Frequency Correction")
plt.grid(True)

# Constellation AFTER correction
plt.subplot(1, 2, 2)
plt.plot(np.real(samples_after), np.imag(samples_after), 'o')
plt.title("Constellation AFTER Frequency Correction")
plt.grid(True)

# Eye diagrams
plt.figure(figsize = (10, 10))
plt.subplot(2, 1, 1)
plot_eye(y, sps, "Eye Diagram BEFORE Frequency Correction")
plt.subplot(2, 1, 2)
plot_eye(y_corrected, sps, "Eye Diagram AFTER Frequency Correction")

plt.show()