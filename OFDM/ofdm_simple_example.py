import numpy as np

symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])

ofdm_time = np.fft.ifft(symbols)
print("ofdm_time", ofdm_time)
# adding a cyclic prefix
cp_length = 1
print("a:", ofdm_time[-cp_length:])
print("b:", ofdm_time)
ofdm_with_cp = np.concatenate([ofdm_time[-cp_length:], ofdm_time])
print("OFDM with cyclic prefix:", ofdm_with_cp)

received = np.fft.fft(ofdm_time)
print("Received:", received)