import numpy as np
from scipy.special import hermite

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
    noise_var = 1 / (2 * snr_lin)
    noise = np.sqrt(noise_var) * np.random.randn(*signal.shape)
    return signal + noise
