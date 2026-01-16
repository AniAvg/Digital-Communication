import matplotlib.pyplot as plt
from scipy.special import erfc
from utils import *

num_bits = 5000
sps = 8
beta = 0.35
span = 18
Ts = 1
snr_db = np.arange(-2, 11, 1)

time = np.arange(-span/2, span/2 + 1/sps, 1/sps)

rrc = root_raised_cosine(time, Ts, beta)
rrc /= np.sqrt(np.sum(rrc ** 2))

herm = hermite_pulse(time, order=10, Ts=Ts)
herm /= np.sqrt(np.sum(herm ** 2))

def simulate_ber(pulse):
    ber = []
    delay = (len(pulse) - 1) // 2
    total_delay = 2 * delay

    for snr in snr_db:
        bits = np.random.randint(0, 2, num_bits)
        symbols = 2 * bits - 1

        upsampled = np.zeros(len(symbols) * sps)
        upsampled[::sps] = symbols

        tx = np.convolve(upsampled, pulse)
        tx_noisy = add_awgn(tx, snr)

        matched_filter = pulse[::-1]
        rx = np.convolve(tx_noisy, matched_filter)

        sample_idx = total_delay + np.arange(len(symbols)) * sps
        samples = rx[sample_idx]


        detected = np.sign(samples)
        detected[detected == 0] = 1

        bit_errors = np.sum(detected != symbols)
        ber.append(bit_errors / num_bits)

    return np.array(ber)

ber_rrc = simulate_ber(rrc)
ber_herm = simulate_ber(herm)

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
