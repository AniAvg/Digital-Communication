import numpy as np
import matplotlib.pyplot as plt

num_bits = 10
sps = 8
beta = 0.35
span = 6
Ts = 1

bits = np.random.randint(0, 2, num_bits)
symbols = 2 * bits - 1

time = np.arange(-span/2, span/2 + 1/sps, 1/sps)

unsampled = np.zeros(len(symbols) * sps)
unsampled[::sps] = symbols

# def raised_cosine(t, Ts, beta):
#     rc = np.zeros_like(t)
#     for i in range(len(t)):
#         if abs(1 - (2 * beta * t[i] / Ts) ** 2) < 1e-6:
#             rc[i] = np.pi / 4 * np.sinc(1 / (2 * beta))
#         else:
#             rc[i] = np.sinc(t[i] / Ts) * \
#                     np.cos(np.pi * beta * t[i] / Ts) / \
#                     (1 - (2 * beta * t[i] / Ts) ** 2)
#     return rc

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

def add_awgn(signal, snr_db):
    snr_lin = 10 ** (snr_db / 10)
    power = np.mean(signal ** 2)
    noise_power = power / snr_lin
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise


rrc_filter = root_raised_cosine(time, Ts, beta)

tx = np.convolve(unsampled, rrc_filter)

tx_with_noise = add_awgn(tx, 5)

matched_filter = rrc_filter[::-1]
y = np.convolve(tx_with_noise, matched_filter)

delay = (len(rrc_filter) - 1) // 2
total_delay = 2 * delay

sample_indices = total_delay + np.arange(len(symbols)) * sps
samples = y[sample_indices]
detected = np.sign(samples)


print("Symbols:  ", symbols)
print("Samples:  ", samples)
print("Detected: ", detected)


def plot_eye(signal, sps, num_symbols=100, offset=0, title="Eye Diagram"):
    eye_len = 2 * sps
    plt.figure(figsize = (8, 4))

    for i in range(num_symbols):
        start = i * sps + offset
        if start + eye_len < len(signal):
            segment = signal[start:start + eye_len]
            plt.plot(segment, color = 'blue', alpha = 0.2)

    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)


plt.figure(figsize = (10, 4))
plt.plot(time, rrc_filter)
plt.title("RRC Pulse")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

tx_time = np.arange(len(tx)) / sps
tx_time_noisy = np.arange(len(tx_with_noise)) / sps

plt.figure(figsize = (10, 9))
plt.subplot(2, 1, 1)
plt.plot(tx_time, tx)
plt.title("Transmit Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(tx_time_noisy, tx_with_noise)
plt.title("Transmit Signal with Noise")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)


y_time = np.arange(len(y)) / sps

plt.figure(figsize = (10, 4))
plt.plot(y_time, y)
plt.stem(sample_indices / sps, samples, linefmt='r-', markerfmt='ro', basefmt=' ', label="Samples")
plt.title("Matched filter Output and Samples")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plot_eye(y, sps, num_symbols=50, offset=delay, title="Eye Diagram (Matched Filter Output)")
plot_eye(tx, sps, num_symbols=50, title="Eye Diagram (Transmit Signal)")
plot_eye(tx_with_noise, sps, num_symbols=50, title="Eye Diagram (Transmit Signal with Noise)")

plt.tight_layout()
plt.show()