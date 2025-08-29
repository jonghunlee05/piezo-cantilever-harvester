import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

print("🔍 Starting sweep + summary analysis...")

# ------------------------------------------------------------------
# System matrices (replace with FEM results)
# ------------------------------------------------------------------
M = np.array([[1.0]])
C = np.array([[0.01]])
K = np.array([[100.0]])

# Piezo coupling
Theta = np.array([[0.5]])   # electromechanical coupling
Cp = np.array([[1e-6]])     # capacitance

ndof = M.shape[0]

# ------------------------------------------------------------------
# Sweep settings
# ------------------------------------------------------------------
excitation_type = "sin"   # only sin supported for sweeps
a0 = 1.0                  # amplitude [m/s^2]

frequencies = np.linspace(5, 50, 10)      # Hz
resistances = np.logspace(3, 6, 5)        # Ohm

t_span = (0, 1.0)
t_eval = np.linspace(t_span[0], t_span[1], 3000)

# Output directory
outdir = "sweep_summary"
os.makedirs(outdir, exist_ok=True)

# ------------------------------------------------------------------
# Excitation function
# ------------------------------------------------------------------
def make_base_accel(f_base):
    omega = 2*np.pi*f_base
    def func(t):
        return a0 * np.sin(omega * t)
    return func

# ------------------------------------------------------------------
# ODE system
# ------------------------------------------------------------------
def make_odes(R_load, base_accel):
    def odes(t, y):
        u = y[:ndof]
        v = y[ndof:2*ndof]
        V = y[-1]

        f_eff = -M @ np.ones(ndof) * base_accel(t)
        u_ddot = np.linalg.solve(M, f_eff - C @ v - K @ u + Theta.T * V)

        I = Theta @ v
        V_dot = -(1.0/R_load) * V / Cp - I / Cp
        dydt = np.concatenate([v.flatten(), u_ddot.flatten(), [V_dot.item()]])
        return dydt
    return odes

# ------------------------------------------------------------------
# Run one case
# ------------------------------------------------------------------
def run_case(f_base, R_load, case_dir):
    os.makedirs(case_dir, exist_ok=True)

    base_accel = make_base_accel(f_base)
    odes = make_odes(R_load, base_accel)

    y0 = np.zeros(2*ndof + 1)
    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method="RK45")

    u = sol.y[0]; v = sol.y[1]; V = sol.y[2]
    I = np.zeros_like(V); P = np.zeros_like(V)
    for i in range(len(sol.t)):
        I[i] = (Theta @ np.array([[v[i]]])).item()
        P[i] = (V[i]**2) / R_load

    dt = sol.t[1] - sol.t[0]
    E = np.cumsum(P) * dt

    # Save CSV
    results_data = np.column_stack([sol.t, u, v, V, I, P, E])
    np.savetxt(
        os.path.join(case_dir, "results.csv"),
        results_data,
        header="Time_s,Displacement_m,Velocity_ms,Voltage_V,Current_A,Power_W,Energy_J",
        delimiter=",",
    )

    return {
        "f_Hz": f_base,
        "R_Ohm": R_load,
        "u_max": np.max(np.abs(u)),
        "V_max": np.max(np.abs(V)),
        "P_max": np.max(P),
        "E_tot": E[-1],
    }

# ------------------------------------------------------------------
# Sweep loop
# ------------------------------------------------------------------
summary = []
for f in frequencies:
    for R in resistances:
        case_name = f"f{f:.1f}Hz_R{R:.1e}"
        case_dir = os.path.join(outdir, case_name)
        print(f"▶ Running case {case_name} ...")
        res = run_case(f, R, case_dir)
        summary.append(res)

# ------------------------------------------------------------------
# Save summary CSV
# ------------------------------------------------------------------
import pandas as pd
df = pd.DataFrame(summary)
df.to_csv(os.path.join(outdir, "sweep_summary.csv"), index=False)
print("💾 Saved sweep_summary.csv")

# ------------------------------------------------------------------
# Plots: Power vs Frequency for each R
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for R in resistances:
    mask = np.isclose(df["R_Ohm"], R)
    plt.plot(df[mask]["f_Hz"], df[mask]["P_max"], marker="o", label=f"R={R:.0e}Ω")
plt.xlabel("Frequency [Hz]"); plt.ylabel("Max Power [W]")
plt.title("Power vs Frequency (different R_load)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(outdir, "power_vs_freq.png"), dpi=300, bbox_inches="tight")

# ------------------------------------------------------------------
# Heatmap: Max Power vs f & R
# ------------------------------------------------------------------
pivot = df.pivot(index="R_Ohm", columns="f_Hz", values="P_max")
plt.figure(figsize=(10, 6))
plt.imshow(pivot, aspect="auto", origin="lower",
           extent=[frequencies.min(), frequencies.max(),
                   resistances.min(), resistances.max()],
           cmap="viridis")
plt.colorbar(label="Max Power [W]")
plt.yscale("log")
plt.xlabel("Frequency [Hz]"); plt.ylabel("Resistance [Ω]")
plt.title("Heatmap: Max Power vs Frequency & Resistance")
plt.savefig(os.path.join(outdir, "power_heatmap.png"), dpi=300, bbox_inches="tight")

print("✅ Sweep summary complete! See:", outdir)
