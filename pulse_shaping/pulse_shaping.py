import numpy as np
import matplotlib.pyplot as plt

num_symbols = 8
sps = 50
alpha = 0.35
Ts = 1
t = np.arange(-4*Ts, 4*Ts, Ts/sps)

symbols = np.random.choice([-1, 1], num_symbols)


rect_pulse = np.ones(sps)

tx_rect = np.zeros(num_symbols * sps)
for i, sym in enumerate(symbols):
    tx_rect[i*sps:(i+1)*sps] = sym * rect_pulse

def raised_cosine(t, Ts, alpha):
    rc = np.zeros_like(t)
    for i in range(len(t)):
        if abs(1 - (2*alpha*t[i]/Ts)**2) < 1e-6:
            rc[i] = np.pi/4 * np.sinc(1/(2*alpha))
        else:
            rc[i] = np.sinc(t[i]/Ts) * \
                    np.cos(np.pi*alpha*t[i]/Ts) / \
                    (1 - (2*alpha*t[i]/Ts)**2)
    return rc

rc_pulse = raised_cosine(t, Ts, alpha)

# RC Pulse Shaping
tx_rc = np.zeros(num_symbols * sps)

for i, sym in enumerate(symbols):
    start = i * sps
    end = start + len(rc_pulse)

    if end <= len(tx_rc):
        tx_rc[start:end] += sym * rc_pulse

time = np.arange(len(tx_rect)) / sps

plt.figure(figsize=(10, 5))

plt.plot(time, tx_rect, label="Without Pulse Shaping")
plt.plot(time, tx_rc, label="With Pulse Shaping (RC)", linewidth=2)

# Symbol sampling instants
for i in range(num_symbols):
    plt.axvline(i, linestyle="--", alpha=0.3)

plt.xlabel("Time (symbols)")
plt.ylabel("Amplitude")
plt.title("Without vs With Raised-Cosine Pulse Shaping")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
