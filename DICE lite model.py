import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =========================
# 1. DICE-lite Parameters
# =========================
@dataclass
class ParamsDICE:
    # Economic Params (Same as SDH)
    T: int = 40
    dt: float = 5.0
    alpha: float = 0.3      # Capital elasticity (gamma in paper)
    g_A: float = 0.01       # TFP growth
    rho: float = 0.02       # Pure time preference
    eta: float = 1.5        # CRRA
    delta_K: float = 0.03   # Depreciation
    
    # Initial States
    K0: float = 10.0
    A0: float = 1.0
    L0: float = 1.0         # Normalized population
    Z0: float = 0.0         # Initial Climate Stock
    
    # DICE Specifics
    sigma_0: float = 0.08   # Initial Emissions Intensity (E/Y)
    g_sigma: float = -0.01  # Decarbonization of tech (exogenous)
    
    psi: float = 0.005      # Damage coefficient (D = psi * Z^2)
    
    kappa: float = 0.08     # Abatement cost scale
    phi_cost: float = 2.4   # Abatement convexity (typically 2.6 in DICE)
    
    theta: float = 0.2      # Emissions to Stock conversion
    phi_Z: float = 0.005     # Natural decay of Z

    def discount_factor(self):
        return np.exp(-self.rho * self.dt)

# =========================
# 2. Smooth Simulation
# =========================
def get_gross_output(K, A, L, alpha):
    K_safe = np.maximum(K, 1e-6)
    return A * (K_safe**alpha) * (L**(1-alpha))

def get_damage_fraction(Z, psi):
    # D(Z) = psi * Z^2
    # We clip it at 0.99 to prevent the world from destroying itself numerically
    damage = psi * (Z**2)
    return np.minimum(damage, 0.99)

def get_abatement_cost_fraction(m, kappa, phi_cost):
    # Phi(m) = kappa * m^phi
    m_safe = np.clip(m, 0, 1) # Force m between 0 and 1 logically
    return kappa * (m_safe**phi_cost)

def simulate_dice_smooth(C, m, params):
    T = params.T
    dt = params.dt
    
    K = np.zeros(T + 1)
    Z = np.zeros(T + 1)
    A = np.zeros(T + 1)
    sigma = np.zeros(T + 1)
    
    K[0] = params.K0
    Z[0] = params.Z0
    A[0] = params.A0
    sigma[0] = params.sigma_0
    
    Y_gross_path = np.zeros(T)
    Y_net_path = np.zeros(T)
    Damages_path = np.zeros(T)
    Abatement_path = np.zeros(T)
    
    for t in range(T):
        # Exogenous Trends
        A[t+1] = A[t] * np.exp(params.g_A * dt)
        sigma[t+1] = sigma[t] * np.exp(params.g_sigma * dt)
        
        # 1. Gross Output
        Y_gross = get_gross_output(K[t], A[t], params.L0, params.alpha)
        Y_gross_path[t] = Y_gross
        
        # 2. Climate Damage & Abatement Cost
        dam_frac = get_damage_fraction(Z[t], params.psi)
        abate_frac = get_abatement_cost_fraction(m[t], params.kappa, params.phi_cost)
        
        Damages_path[t] = dam_frac
        Abatement_path[t] = abate_frac
        
        # 3. Net Output
        # Y_net = (1 - D)(1 - Abate) * Y_gross
        Y_net = (1 - dam_frac) * (1 - abate_frac) * Y_gross
        Y_net_path[t] = Y_net
        
        # 4. Capital Motion
        # K_next = K + (Y_net - C) - dep
        investment = Y_net - C[t]
        K[t+1] = (1 - params.delta_K * dt) * K[t] + investment * dt
        
        # 5. Emissions & Climate
        # E = sigma * (1 - m) * Y_gross
        emissions = sigma[t] * (1 - m[t]) * Y_gross
        
        # Z_next = (1 - phi)*Z + theta*E
        Z[t+1] = (1 - params.phi_Z * dt) * Z[t] + params.theta * emissions * dt
        
    return K, Z, Y_net_path, Y_gross_path, Damages_path, Abatement_path

# =========================
# 3. Objective with Terminal Value
# =========================
def u_crra(c, eta):
    c_safe = np.maximum(c, 1e-6)
    if abs(eta - 1.0) < 1e-8: return np.log(c_safe)
    else: return (c_safe**(1.0 - eta) - 1.0) / (1.0 - eta)

def dicelite_objective(z_vector, params):
    T = params.T
    dt = params.dt
    beta = params.discount_factor()
    
    C = z_vector[:T]
    m = z_vector[T:] # Control is now m (0 to 1)
    
    K, Z, Y_net, Y_gross, D_path, A_path = simulate_dice_smooth(C, m, params)
    
    # --- UTILITY ---
    utility = 0.0
    df = 1.0
    for t in range(T):
        utility += df * u_crra(C[t], params.eta)
        df *= beta
        
    # --- TERMINAL VALUE (Critically important here) ---
    # In DICE, Z reduces output capacity. Terminal value must reflect this.
    
    # 1. Sustainable Gross Output
    K_final = np.maximum(K[-1], 1e-1)
    Y_final_gross = get_gross_output(K_final, params.A0 * np.exp(params.g_A * T * dt), params.L0, params.alpha)
    
    # 2. Climate Drag at T
    # If we leave high Z, our future capacity is permanently damaged.
    Z_final = np.maximum(Z[-1], 0.0)
    dam_final = get_damage_fraction(Z_final, params.psi)
    
    # 3. Sustainable Net Output
    # Assume we maintain the final mitigation rate
    m_final = np.clip(m[-1], 0, 1)
    abate_final = get_abatement_cost_fraction(m_final, params.kappa, params.phi_cost)
    
    Y_final_net = (1 - dam_final) * (1 - abate_final) * Y_final_gross
    
    # 4. Sustainable Consumption
    C_sust = np.maximum(Y_final_net - params.delta_K * K_final, 1e-3)
    
    # Extension multiplier
    multiplier = 0.0
    df_term = df
    for _ in range(10): 
        multiplier += df_term
        df_term *= beta
        
    term_val = multiplier * u_crra(C_sust, params.eta)
    total_welfare = utility + term_val
    
    # --- PENALTIES ---
    penalty = 0.0
    scale = 1e5
    
    # Non-negative Capital
    penalty += scale * np.sum(np.minimum(K, 0.0)**2)
    # Non-negative Consumption
    penalty += scale * np.sum(np.minimum(C, 0.0)**2)
    # m must be roughly 0 to 1 (bounds handle this mostly, but soft penalty helps gradients)
    penalty += scale * np.sum(np.maximum(m - 1.0, 0.0)**2)
    penalty += scale * np.sum(np.minimum(m, 0.0)**2)

    return -(total_welfare - penalty)

# =========================
# 4. Initial Guess
# =========================
def get_initial_guess_dice(params):
    T = params.T
    
    # 1. Mitigation starts at 0, ramps up to 0.5 (typical IAM path)
    m_guess = np.linspace(0.0, 0.5, T)
    
    # 2. Consumption guess
    # Run a dummy sim to get output
    K_dummy = np.zeros(T)
    K_curr = params.K0
    A_curr = params.A0
    C_guess = np.zeros(T)
    
    for t in range(T):
        Y = get_gross_output(K_curr, A_curr, params.L0, params.alpha)
        C_guess[t] = 0.75 * Y # 75% rule
        # Simple update
        K_curr = (1 - params.delta_K*params.dt)*K_curr + (Y - C_guess[t])*params.dt
        A_curr *= np.exp(params.g_A * params.dt)
        
    return np.concatenate([C_guess, m_guess])

# =========================
# 5. Execution
# =========================
if __name__ == "__main__":
    params = ParamsDICE()
    print("Solving DICE-lite Model (Output Damages)...")
    
    z0 = get_initial_guess_dice(params)
    
    # Bounds
    # C > 0
    # m in [0, 1] (Strict bounds for mitigation rate)
    bounds = [(1e-3, None) for _ in range(params.T)] + \
             [(0.0, 1.0) for _ in range(params.T)]
    
    result = minimize(
        dicelite_objective,
        z0,
        args=(params,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 20000, 'ftol': 1e-9, 'disp': True}
    )
    
    print(f"\nOptimization Success: {result.success}")
    
    # Extract
    C_opt = result.x[:params.T]
    m_opt = result.x[params.T:]
    
    K, Z, Y_net, Y_gross, D_path, A_path = simulate_dice_smooth(C_opt, m_opt, params)
    
    print("\n--- DICE-lite Diagnostics ---")
    print(f"Final Capital: {K[-1]:.2f}")
    print(f"Final Climate Stock (Z): {Z[-1]:.2f}")
    print(f"Final Damage Fraction: {D_path[-1]*100:.2f}% of Gross Output")
    print(f"Final Mitigation Rate (m): {m_opt[-1]*100:.2f}%")
    
    c_growth = (C_opt[-1]/C_opt[0] - 1)*100
    print(f"Consumption Growth: {c_growth:.2f}%")

    # Plot
    t = np.arange(params.T) * params.dt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    
    ax[0,0].plot(t, C_opt, color='green')
    ax[0,0].set_title("Consumption")
    
    ax[0,1].plot(t, m_opt*100, color='blue')
    ax[0,1].set_title("Mitigation Rate (%)")
    ax[0,1].set_ylim(0, 100)
    
    ax[0,2].plot(t, Y_gross, 'k--', label='Gross')
    ax[0,2].plot(t, Y_net, 'k-', label='Net')
    ax[0,2].set_title("Output (Gross vs Net)")
    ax[0,2].legend()
    
    ax[1,0].plot(np.arange(params.T+1)*params.dt, K, color='purple')
    ax[1,0].set_title("Capital Stock")
    
    ax[1,1].plot(np.arange(params.T+1)*params.dt, Z, color='red')
    ax[1,1].set_title("Climate Stock (Z)")
    
    ax[1,2].plot(t, D_path*100, label='Damages', color='red')
    ax[1,2].plot(t, A_path*100, label='Abatement Cost', color='blue')
    ax[1,2].set_title("Costs (% of Output)")
    ax[1,2].legend()
    
    for a in ax.flatten(): a.grid(True)
    plt.tight_layout()
    plt.show()