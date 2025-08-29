import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

print("🔍 Starting transient time-domain simulation...")

# ------------------------------------------------------------------
# Example system matrices (replace with your assembled FEM matrices)
# ------------------------------------------------------------------
M = np.array([[1.0]])
C = np.array([[0.01]])
K = np.array([[100.0]])

# Piezo coupling
Theta = np.array([[0.5]])   # electromechanical coupling
Cp = np.array([[1e-6]])     # capacitance
R_load = 1e5                # load resistance [ohm]

ndof = M.shape[0]

# ------------------------------------------------------------------
# Excitation settings
# ------------------------------------------------------------------
excitation_type = "sin"   # options: "sin", "step", "random"
f_base = 20.0             # Hz (for sinusoidal)
omega = 2*np.pi*f_base
a0 = 1.0                  # amplitude [m/s^2]
rng = np.random.default_rng(seed=42)  # reproducible random

def base_accel(t):
    """Return base acceleration depending on excitation type."""
    if excitation_type == "sin":
        return a0 * np.sin(omega * t)
    elif excitation_type == "step":
        return a0 if t > 0.1 else 0.0  # step starts at 0.1 s
    elif excitation_type == "random":
        return a0 * rng.normal(0, 1)   # white noise, scaled
    else:
        raise ValueError(f"Unknown excitation_type: {excitation_type}")

# ------------------------------------------------------------------
# State-space form
# ------------------------------------------------------------------
def odes(t, y):
    u = y[:ndof]
    v = y[ndof:2*ndof]
    V = y[-1]

    # Effective force from base acceleration
    f_eff = -M @ np.ones(ndof) * base_accel(t)

    # Mechanical acceleration
    u_ddot = np.linalg.solve(M, f_eff - C @ v - K @ u + Theta.T * V)

    # Electrical dynamics
    I = Theta @ v
    V_dot = -(1.0/R_load) * V / Cp - I / Cp

    dydt = np.concatenate([v.flatten(), u_ddot.flatten(), [V_dot.item()]])
    return dydt

# ------------------------------------------------------------------
# Initial conditions & integration
# ------------------------------------------------------------------
y0 = np.zeros(2*ndof + 1)
t_span = (0, 1.0)  # 1 second
t_eval = np.linspace(t_span[0], t_span[1], 5000)

sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method="RK45")

# Extract solution components
u = sol.y[0]
v = sol.y[1]
V = sol.y[2]

# Current and power
I = np.zeros_like(V)
P = np.zeros_like(V)
for i in range(len(sol.t)):
    I[i] = (Theta @ np.array([[v[i]]])).item()
    P[i] = (V[i]**2) / R_load

# ------------------------------------------------------------------
# Energy & FFT
# ------------------------------------------------------------------
# Energy harvested (Joules) = integral of power over time
try:
    from scipy.integrate import cumtrapz
    E = cumtrapz(P, sol.t, initial=0)
except ImportError:
    # Fallback: simple cumulative sum approximation
    dt = sol.t[1] - sol.t[0]
    E = np.cumsum(P) * dt

plt.figure(figsize=(10, 6))
plt.plot(sol.t, E, label="Cumulative Energy [J]", linewidth=2, color="purple")
plt.xlabel("Time [s]"); plt.ylabel("Energy [J]")
plt.title("Piezoelectric Cantilever - Harvested Energy")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("transient_energy.png", dpi=300, bbox_inches="tight")

dt = sol.t[1] - sol.t[0]
N = len(sol.t)
freqs = np.fft.rfftfreq(N, dt)
U_fft = np.abs(np.fft.rfft(u))
V_fft = np.abs(np.fft.rfft(V))

plt.figure(figsize=(10, 6))
plt.semilogy(freqs, U_fft, label="|FFT(u)| Displacement")
plt.semilogy(freqs, V_fft, label="|FFT(V)| Voltage")
plt.xlabel("Frequency [Hz]"); plt.ylabel("Amplitude (log scale)")
plt.title("Piezoelectric Cantilever - FFT Spectrum")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("transient_fft.png", dpi=300, bbox_inches="tight")

# ------------------------------------------------------------------
# Standard plots
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(sol.t, u, label="Displacement [m]", linewidth=2)
plt.xlabel("Time [s]"); plt.ylabel("Displacement [m]")
plt.title("Transient Displacement"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("transient_displacement.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(10, 6))
plt.plot(sol.t, V, label="Voltage [V]", linewidth=2, color="orange")
plt.xlabel("Time [s]"); plt.ylabel("Voltage [V]")
plt.title("Transient Voltage"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("transient_voltage.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(10, 6))
plt.plot(sol.t, I, label="Current [A]", linewidth=2, color="green")
plt.xlabel("Time [s]"); plt.ylabel("Current [A]")
plt.title("Transient Current"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("transient_current.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(10, 6))
plt.plot(sol.t, P, label="Power [W]", linewidth=2, color="red")
plt.xlabel("Time [s]"); plt.ylabel("Power [W]")
plt.title("Transient Power"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("transient_power.png", dpi=300, bbox_inches="tight")

# ------------------------------------------------------------------
# Save results to CSV
# ------------------------------------------------------------------
results_data = np.column_stack([sol.t, u, v, V, I, P, E])
np.savetxt(
    "transient_results.csv",
    results_data,
    header="Time_s,Displacement_m,Velocity_ms,Voltage_V,Current_A,Power_W,Energy_J",
    delimiter=",",
)

print("✅ Transient simulation complete!")
print(f"  - Excitation type: {excitation_type}")
print(f"  - Simulation time: {t_span[1]:.1f} s")
print(f"  - Time steps: {len(sol.t)}")
print(f"  - Max displacement: {np.max(np.abs(u)):.3e} m")
print(f"  - Max voltage: {np.max(np.abs(V)):.3e} V")
print(f"  - Max power: {np.max(P):.3e} W")
print(f"  - Total harvested energy: {E[-1]:.3e} J")
