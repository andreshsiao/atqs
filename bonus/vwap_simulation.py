import numpy as np
import matplotlib.pyplot as plt


# ---------- (i) Simulate cumulative market volume V_t ----------
T = 1.0
N = 100
dt = T / N
times = np.linspace(0, T, N + 1)

# U-shape profile (scaled to [0.5, 1.5] for illustration)
u_profile = 0.5 + np.abs(np.cos(np.pi * times))**4  # Now in [0.5, 1.5]
u_profile = u_profile / np.sum(u_profile * dt)  # Normalize integral to 1

# Add noise and clip
inst_vol = u_profile + 0.02 * np.random.randn(N + 1)
inst_vol = np.clip(inst_vol, 0, None)

# Cumulative volume (before normalization)
V_t_unnormalized = np.cumsum(inst_vol) * dt
inst_vol = np.diff(V_t_unnormalized, prepend=0) / dt  # Proper derivative
V_t = V_t_unnormalized / V_t_unnormalized[-1]  # Normalize to [0, 1]

# Plot
plt.figure(figsize=(18, 5))
plt.subplot(1, 2, 1)
plt.plot(times, inst_vol, label="Instantaneous Volume - U Shape", lw=2)
plt.xlabel("Time")
plt.ylabel("Volume")
plt.title("Instantaneous Volume (Properly Scaled)")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(times, V_t, label="Cumulative Volume", lw=2)
plt.xlabel("Time")
plt.ylabel("Cumulative Volume")
plt.title("Cumulative Volume (Normalized to [0, 1])")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("vwap_volume.png")
plt.show()

# ---------- (ii) Parameters ----------
sigma = 0.02
b = 0.01
k = 0.10
phi = 1
alpha = 0.5
rho = 0.5
Q0 = 1.0

xi = np.sqrt(phi * (1 - rho) ** 2 / k)
gamma = (alpha - 0.5 * b + np.sqrt(k * phi)*(1 - rho)) / (alpha - 0.5 * b - np.sqrt(k * phi) * (1 - rho))
print("xi:", xi)
print("gamma:", gamma)
# ---------- (iii) Helper for ell(u, t) ----------
def ell(u, t, T, xi, gamma):
    num = gamma * np.exp(xi * (T - u)) - np.exp(-xi * (T - u))
    den = gamma * np.exp(xi * (T - t)) - np.exp(-xi * (T - t))
    return num / den

# ---------- (iv) Precompute h2(t) ----------
h2 = (
    -np.sqrt(k * phi) * (1 - rho)
    * (gamma * np.exp(xi * (T - times)) + np.exp(-xi * (T - times)))
    / (gamma * np.exp(xi * (T - times)) - np.exp(-xi * (T - times)))
    - 0.5 * b
)

# ---------- (v) Precompute h1(t, V_t) ----------
h1 = np.zeros_like(times)
for i, t in enumerate(times):
    integral = 0.0
    for j in range(i, len(times)):
        u = times[j]
        expected_Vu = V_t[j]
        integrand = ell(u, t, T, xi, gamma) * ((1 - rho) * Q0 - rho * expected_Vu)
        integral += integrand * dt
    h1[i] = 2 * phi * integral

# ---------- (vi) Simulate optimal trading rate and inventory path ----------
Q = np.zeros_like(times)
Q[0] = Q0
nu_star = np.zeros_like(times)

for i, t in enumerate(times):
    q = Q[i]
    nu_star[i] = -(b * q + h1[i] + 2 * h2[i] * q) / (2 * k)
    nu_star[i] = max(nu_star[i], 0.0)
    if i < N:
        Q[i + 1] = Q[i] - nu_star[i] * dt

# ---------- (vii) Combined Plot: nu_star, Q, V_t ----------
plt.figure(figsize=(18, 5))

# (1) Optimal trading speed
plt.subplot(1, 3, 1)
plt.plot(times, nu_star, label=r'Optimal rate $\nu_t^*$')
plt.xlabel("Time")
plt.ylabel("Trading speed")
plt.title("Optimal Trading Rate")
plt.grid(True)
plt.legend()

# (2) Inventory trajectory
plt.subplot(1, 3, 2)
plt.plot(times, Q, label=r'Inventory $Q_t$', color='tab:orange')
plt.xlabel("Time")
plt.ylabel("Inventory")
plt.title("Inventory Trajectory")
plt.grid(True)
plt.legend()

# (3) Cumulative volume V_t
plt.subplot(1, 3, 3)
plt.plot(times, V_t, label=r'Cumulative Volume $V_t$', color='tab:green')
plt.xlabel("Time")
plt.ylabel("Volume")
plt.title("Simulated Cumulative Volume $V_t$")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("vwap_rate_inventory.png")
plt.show()
