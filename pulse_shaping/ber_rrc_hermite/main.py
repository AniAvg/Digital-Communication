import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from functions import *

num_bits = 5000
sps = 8
beta = 0.35
span = 16
Ts = 1
snr_db = np.arange(0, 16, 1)


bits = np.random.randint(0, 2, num_bits)
symbols = 2 * bits - 1

upsampled = np.zeros(len(symbols)  * sps)
upsampled[::sps] = symbols

time = np.arange(-span / 2, span / 2 + 1 / sps, 1 / sps)

rrc = root_raised_cosine(time, Ts, beta)
rrc /= np.sqrt(np.sum(rrc ** 2))

herm = hermite_pulse(time, order=0, Ts=Ts)

def simulate_ber(pulse, upsampled):
    ber = []

    tx = np.convolve(upsampled, pulse)
    delay = len(pulse) - 1

    for snr in snr_db:
        rx = np.convolve(add_awgn(tx, snr, sps), pulse[::-1])

        sample_idx = delay + np.arange(num_bits) * sps

        # discard filter transients
        discard = span
        valid_idx = sample_idx[discard:-discard]
        valid_bits = bits[discard:-discard]

        detected = np.sign(rx[valid_idx])
        errors = np.sum(valid_bits != detected)

        ber.append(errors / len(valid_bits))

    return np.array(ber)
#
# def simulate_rrc(time, Ts, beta, unsampled, symbols, bits):
#     snr_list = np.arange(0, 16, 1)
#     ber_list = []
#
#     for snr in snr_list:
#         errors = 0
#         total_bits = 0
#         for _ in range(1000):
#             tx_rrc = np.convolve(unsampled, rrc_filter)
#             tx_rrc_noisy = add_awgn(tx_rrc, snr, sps)##
#             y = np.convolve(tx_rrc_noisy, rrc_filter[::-1])
#
#             delay = (len(rrc_filter) - 1) // 2
#             total_delay = 2 * delay
#
#             sample_indices = total_delay + np.arange(len(symbols)) * sps
#             samples = y[sample_indices]
#             detected = np.sign(samples)
#
#             errors += np.sum(bits != detected)
#             total_bits += len(bits)
#         ber = errors / total_bits if total_bits > 0 else 1
#         ber_list.append(ber)
#         print(f"RRC  SNR={snr:2d} dB → BER={ber:.2e}")
#     return snr_list, ber_list
#
# def simulate_hermite(time, Ts, unsampled, symbols, bits):
#     snr_list = np.arange(0, 16, 1)
#     ber_list = []
#
#     hermite_filter = hermite_pulse(time, 0, Ts)
#     hermite_filter /= np.sqrt(np.sum(hermite_filter ** 2))
#     for snr in snr_list:
#         errors = 0
#         total_bits = 0
#         for _ in range(1000):
#             tx_herm = np.convolve(unsampled, hermite_filter)
#             tx_herm_noisy = add_awgn(tx_herm, snr, sps)###
#             y = np.convolve(tx_herm_noisy, hermite_filter[::-1])
#
#             delay = (len(hermite_filter) - 1) // 2
#             total_delay = 2 * delay
#
#             sample_indices = total_delay + np.arange(len(symbols)) * sps
#             samples = y[sample_indices]
#             detected = np.sign(samples)
#
#             errors += np.sum(bits != detected)
#             total_bits += len(bits)
#         ber = errors / total_bits
#         ber_list.append(ber)
#         print(f"Hermite  SNR={snr:2d} dB → BER={ber:.2e}")
#     return snr_list, ber_list

# snr_rrc, ber_rrc = simulate_rrc(time, Ts, beta, unsampled, symbols, bits)
# snr_herm, ber_herm = simulate_hermite(time, Ts, unsampled, symbols, bits)

ber_rrc = simulate_ber(rrc, upsampled)
ber_herm = simulate_ber(herm, upsampled)

snr_lin = 10 ** (snr_db / 10)
#Theoretical BER for BPSK
ber_theory = 0.5 * erfc(np.sqrt(snr_lin))

plt.figure(figsize = (8, 5))
plt.semilogy(snr_db, ber_rrc, 'o-', label = "RRC")
plt.semilogy(snr_db, ber_herm, 's-', label = "Hermite")
plt.semilogy(snr_db, ber_theory, 'k--', linewidth = 2, label = "BPSK theory")
plt.grid(True, which = "both")
plt.xlabel("SNR")
plt.ylabel("BER")
plt.title("BER vs SNR")
plt.legend()

plt.ylim(1e-5, 1)
plt.show()


