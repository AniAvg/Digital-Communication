import numpy as np
import matplotlib.pyplot as plt

msg = input("Enter a message: ")
msg_binary = ""

for c in msg:
    binary = format(ord(c), '08b')
    msg_binary += binary

print("Binary message: ", msg_binary)

msg_symbols = np.array(list(msg_binary), dtype = int)

bits_per_symbol = 4

if len(msg_symbols) % 4 != 0:
    pad = 4 - (len(msg_symbols) % 4)
    msg_symbols = np.concatenate((msg_symbols, np.zeros(pad, dtype=int)))

msg_matrix = np.reshape(msg_symbols, (-1, bits_per_symbol))

level_to_bits = {
    -3: [0, 0],
    -1: [0, 1],
     1: [1, 1],
     3: [1, 0]
}
bits_to_level = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1):  1,
    (1, 0):  3
}

qam_symbols = []

for group in msg_matrix:
    i_bits = tuple(group[:2])
    q_bits = tuple(group[2:])
    I = bits_to_level[i_bits]
    Q = bits_to_level[q_bits]
    qam_symbols.append(complex(I, Q))

qam_symbols = np.array(qam_symbols)

print("QAM symbols: ", qam_symbols)

def adding_awgn(symbols, snr_db):
    signal_power = np.mean(np.abs(symbols ** 2))
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    std_dev = np.sqrt(noise_power / 2)
    noise = std_dev * (np.random.randn(len(symbols)) +
                   1j * np.random.randn(len(symbols)))
    noisy_symbols = symbols + noise
    return noisy_symbols

snr_db = 15
qam_noisy_symbols = adding_awgn(qam_symbols, snr_db)

levels = np.array([-3, -1, 1, 3])

received_bits = []
for s in qam_noisy_symbols:
    nearest_I = levels[np.argmin(np.abs(np.real(s) - levels))]
    nearest_Q = levels[np.argmin(np.abs(np.imag(s) - levels))]

    received_bits.extend(level_to_bits[nearest_I])
    received_bits.extend(level_to_bits[nearest_Q])

received_bits = np.array(received_bits, dtype=int)

# Remove padding if added
bit_string_received = ''.join(str(b) for b in received_bits[:len(msg_binary)])

chars = []
for i in range(0, len(bit_string_received), 8):
    byte = bit_string_received[i:i+8]
    chars.append(chr(int(byte, 2)))

recovered_text = ''.join(chars)
print("Recovered text:", recovered_text)


plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
plt.scatter(np.real(qam_symbols), np.imag(qam_symbols), s = 100)
plt.axhline(0, color='gray', linewidth=1.2)
plt.axvline(0, color='gray', linewidth=1.2)
plt.xlim([-3.5, 3.5])
plt.ylim([-3.5, 3.5])
plt.title("16 QAM Constellation Diagram")
plt.xlabel("In phase")
plt.ylabel("Quadrature")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(np.real(qam_noisy_symbols), np.imag(qam_noisy_symbols), s = 100)
plt.axhline(0, color='gray', linewidth=1.2)
plt.axvline(0, color='gray', linewidth=1.2)
plt.xlim([-3.5, 3.5])
plt.ylim([-3.5, 3.5])
plt.title("16 QAM Constellation Diagram with Noise")
plt.xlabel("In phase")
plt.ylabel("Quadrature")
plt.grid(True)

plt.tight_layout()
plt.show()
