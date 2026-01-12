import numpy as np
import matplotlib.pyplot as plt

def qam4_mod(bits):
    bits = bits.reshape(-1, 2)
    mapping = {
        (0, 0): -1-1j,
        (0, 1): -1+1j,
        (1, 0): 1-1j,
        (1, 1): 1+1j
    }
    return np.array([mapping[tuple(b)] for b in bits])

def qam4_demod(symbols):
    all_bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    const_points = np.array([
        -1-1j, -1+1j, 1-1j, 1+1j
    ])

    decoded_bits = []
    for s in symbols:
        idx = np.argmin(np.abs(const_points - s))
        decoded_bits.append(all_bits[idx])
    return np.vstack(decoded_bits).flatten()

def add_awgn(signal, snr_db):
    snr_lin = 10**(snr_db / 10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power / snr_lin
    noise = (np.sqrt(noise_power/2) *
            (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape)))
    return signal + noise

N = 64
cp_len = N // 4
snr_db = 5

bits_number = 10000
bits = np.random.randint(0, 2, size=bits_number)
if len(bits) % 2 != 0:
    bits = np.append(bits, 0)

qam_symbols = qam4_mod(bits)

# Serial to parallel
num_symbols = int(np.ceil(len(qam_symbols) / N))
pad_len = num_symbols * N - len(qam_symbols)
if pad_len > 0:
    symbols = np.concatenate([qam_symbols, np.zeros(pad_len, dtype=np.complex128)])
else:
    symbols = qam_symbols

ofdm_blocks = symbols.reshape(num_symbols, N)

time_domain = np.fft.ifft(ofdm_blocks, n=N, axis=1)

tx_with_cp = np.concatenate([time_domain[:, -cp_len:], time_domain], axis=1)
tx_signal = tx_with_cp.reshape(-1)

rx_signal = add_awgn(tx_signal, snr_db)

rx_with_cp = rx_signal.reshape(num_symbols, N + cp_len)
rx = rx_with_cp[:, cp_len:]
rx_freq = np.fft.fft(rx, n=N, axis=1)

rx_symbols = rx_freq.reshape(-1)[:len(symbols) - pad_len]
rx_bits = qam4_demod(rx_symbols)

tx_bits = bits[:len(rx_bits)]
ber = np.mean(tx_bits != rx_bits)
print(f"BER (SNR={snr_db}): {ber:.5e})")

error = tx_bits - rx_bits
print("Error: ", error)

plt.figure(figsize=(10,4))
plt.plot(np.abs(tx_signal[:4*N]), label="Tx")
plt.plot(np.abs(rx_signal[:4*N]), label="Rx")
plt.title("Time-Domain OFDM Tx and Rx Signals")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

tx_const = ofdm_blocks.flatten()[:2000]
rx_const = rx_freq.flatten()[:2000]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(np.real(tx_const), np.imag(tx_const), s=50)
plt.title("Constellation Before Channel")
plt.xlabel("Real")
plt.ylabel("Imag")
plt.grid(True)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(np.real(rx_const), np.imag(rx_const), s=50, alpha=0.5)
plt.title("Constellation After Channel + Noise")
plt.xlabel("Real")
plt.ylabel("Imag")
plt.grid(True)
plt.axis('equal')

plt.show()