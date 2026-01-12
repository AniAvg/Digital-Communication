import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def generate_bits(n, min_len=100, max_len=2000):
    L = np.random.randint(min_len, max_len + 1)
    L += (-L) % n
    return np.random.randint(0, 2, size=L, dtype=np.uint8)

def ber_qam_theoretical(snr_db, M):
    k = np.log2(M)
    snr_lin = 10**(snr_db / 10)
    return 2 * (1 - 1/np.sqrt(M)) * 0.5 * erfc(np.sqrt((3 * snr_lin) / (2*(M-1))))

# 4 QAM
def qam4_mod(bits):
    bits = bits.reshape(-1, 2)
    mapping = {
        (0, 0): -1-1j,
        (0, 1): -1+1j,
        (1, 0): 1-1j,
        (1, 1): 1+1j
    }
    symbols = np.array([mapping[tuple(b)] for b in bits])
    return symbols / np.sqrt(2)

def qam4_demod(symbols):
    all_bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    const_points = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]) / np.sqrt(2)

    decoded_bits = []
    for s in symbols:
        idx = np.argmin(np.abs(const_points - s))
        decoded_bits.append(all_bits[idx])
    return np.hstack(decoded_bits)

# 16 QAM
def qam16_mod(bits):
    bits = bits.reshape(-1, 4)
    gray_map = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}

    symbols = []
    for b in bits:
        I = gray_map[(b[0], b[1])]
        Q = gray_map[(b[2], b[3])]
        symbols.append(I + 1j * Q)
    symbols = np.array(symbols)
    return symbols / np.sqrt(10)

def qam16_demod(symbols):
    thresh = np.array([-2, 0, 2]) / np.sqrt(10)

    def decide(x):
        if x < thresh[0]: return -3
        if x < thresh[1]: return -1
        if x < thresh[2]: return 1
        return 3

    gray_bits = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}

    bits_out = []
    for s in symbols:
        I = decide(s.real)
        Q = decide(s.imag)
        bits_out.extend(gray_bits[I])
        bits_out.extend(gray_bits[Q])
    return np.array(bits_out, dtype=np.uint8)


def add_awgn(signal, snr_db):
    snr_lin = 10**(snr_db / 10)
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / snr_lin
    noise = (np.sqrt(noise_power / 2) *
            (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)))
    return signal + noise

def simulate(mod, demod, bits_per_symbol, M):
    snr_list = np.arange(0, 36, 5)  # 0 to 35 dB
    ber_list = []
    all_symbols = []

    for snr in snr_list:
        errors = 0
        total_bits = 0
        for _ in range(3000):
            bits = generate_bits(bits_per_symbol)
            tx = mod(bits)
            rx = add_awgn(tx, snr)
            rx_bits = demod(rx)

            if snr == snr_list[0]:
                all_symbols.extend(tx)

            errors += np.sum(bits != rx_bits)
            total_bits += len(bits)

        ber = errors / total_bits if total_bits > 0 else 1
        ber_list.append(ber)
        print(f"{M}-QAM  SNR={snr:2d} dB → BER={ber:.2e}")

    return snr_list, ber_list, np.array(all_symbols)


def simulate_ofdm(mod, demod, bits_per_symbol, N=64, cp_len=16):
    snr_list = np.arange(0, 36, 5)
    ber_list = []
    all_symbols = []
    
    for snr in snr_list:
        errors = 0
        total_bits = 0
        
        for _ in range(300):
            bits = generate_bits(bits_per_symbol, min_len=1000, max_len=2000)
            qam_symbols = mod(bits)
            
            # Serial to parallel
            num_ofdm_symbols = int(np.ceil(len(qam_symbols) / N))
            pad_len = num_ofdm_symbols * N - len(qam_symbols)
            if pad_len > 0:
                symbols = np.concatenate([qam_symbols, np.zeros(pad_len, dtype=np.complex128)])
            else:
                symbols = qam_symbols

            if snr == snr_list[0]:
                all_symbols.extend(qam_symbols)

            ofdm_blocks = symbols.reshape(num_ofdm_symbols, N)
            # IFFT
            time_domain = np.fft.ifft(ofdm_blocks, n=N, axis=1)
            # Add cyclic prefix
            tx_with_cp = np.concatenate([time_domain[:, -cp_len:], time_domain], axis=1)
            tx_signal = tx_with_cp.reshape(-1)
            # Add AWGN
            rx_signal = add_awgn(tx_signal, snr)
            # Remove cyclic prefix
            rx_with_cp = rx_signal.reshape(num_ofdm_symbols, N + cp_len)
            rx = rx_with_cp[:, cp_len:]
            # FFT
            rx_freq = np.fft.fft(rx, n=N, axis=1)

            rx_symbols = rx_freq.reshape(-1)[:len(qam_symbols)]
            rx_bits = demod(rx_symbols)

            tx_bits = bits[:len(rx_bits)]
            errors += np.sum(tx_bits != rx_bits)
            total_bits += len(tx_bits)
        
        ber = errors / total_bits if total_bits > 0 else 1
        ber_list.append(ber)
        print(f"OFDM 4-QAM  SNR={snr:2d} dB → BER={ber:.2e}")
    
    return snr_list, ber_list, np.array(all_symbols)

print("4-QAM Simulation")
snrs_4qam, bers_4qam, symbols_4qam = simulate(qam4_mod, qam4_demod, 2, 4)
print("\n16-QAM Simulation")
snrs_16qam, bers_16qam, symbols_16qam = simulate(qam16_mod, qam16_demod, 4, 16)
print("\nOFDM 4-QAM Simulation")
snrs_ofdm, bers_ofdm, symbols_ofdm = simulate_ofdm(qam4_mod, qam4_demod, 2)

print("\n")
print("Energy Calculations (from simulation data)")

# 4-QAM energies
Es_4qam = np.mean(np.abs(symbols_4qam)**2)
Eb_4qam = Es_4qam / 2

print(f"4-QAM:")
print(f"  Average Symbol Energy (Es) = {Es_4qam:.6f}")
print(f"  Average Bit Energy (Eb) = {Eb_4qam:.6f}")
print(f"  Bits per symbol (k) = 2")
print(f"  Number of symbols analyzed = {len(symbols_4qam)}")

# 16-QAM energies
Es_16qam = np.mean(np.abs(symbols_16qam)**2)
Eb_16qam = Es_16qam / 4

print(f"\n16-QAM:")
print(f"  Average Symbol Energy (Es) = {Es_16qam:.6f}")
print(f"  Average Bit Energy (Eb) = {Eb_16qam:.6f}")
print(f"  Bits per symbol (k) = 4")

snr_fine = np.linspace(0, 35, 200)
ber_4qam_th = ber_qam_theoretical(snr_fine, 4)
ber_16qam_th = ber_qam_theoretical(snr_fine, 16)

plt.figure(figsize = (12, 7))
plt.semilogy(snrs_4qam, bers_4qam, marker='o', label="4-QAM", markersize=8)
plt.semilogy(snrs_16qam, bers_16qam, marker='s', label="16-QAM", markersize=8)
plt.semilogy(snrs_ofdm, bers_ofdm, marker='^', label="OFDM 4-QAM", markersize=8, linewidth=2)
plt.semilogy(snr_fine, ber_4qam_th, '-', label='4-QAM Theoretical', linewidth=2, alpha=0.7)
plt.semilogy(snr_fine, ber_16qam_th, '--', label='16-QAM Theoretical', linewidth=2, alpha=0.7)
plt.xlabel("Eₛ/N₀ (dB)", fontsize=12)
plt.ylabel("BER", fontsize=12)
plt.title("BER vs Eₛ/N₀ (4-QAM, 16-QAM, and OFDM 4-QAM)", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.ylim(1e-7, 1)

plt.show()