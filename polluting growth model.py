import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =========================
# 1. Parameters & Setup
# =========================
@dataclass
class ParamsPolluting:
    # Economic Params (Same as SDH)
    T: int = 40
    dt: float = 5.0
    alpha: float = 0.3
    beta_r: float = 0.05
    g_A: float = 0.01
    rho: float = 0.02
    eta: float = 1.5
    delta_K: float = 0.03
    K0: float = 10.0
    X0: float = 100.0
    A0: float = 1.0

    # Climate Params
    Z0: float = 0.0         # Initial Pollution
    theta: float = 0.5      # Emissions intensity
    phi_Z: float = 0.05     # Natural decay
    chi: float = 0.3        # Abatement tech scale
    nu: float = 0.8         # Abatement concavity
    omega: float = 0.005    # Disutility of pollution

    def discount_factor(self):
        return np.exp(-self.rho * self.dt)

# =========================
# 2. Smooth Simulation (No Clipping)
# =========================
def production(K, x, A, alpha, beta_r):
    K_safe = np.maximum(K, 1e-6)
    x_safe = np.maximum(x, 1e-6)
    return A * (K_safe**alpha) * (x_safe**beta_r)

def abatement_func(q, chi, nu):
    q_safe = np.maximum(q, 1e-6)
    return chi * (q_safe**nu)

def u_poll(c, z, eta, omega):
    c_safe = np.maximum(c, 1e-6)
    # Standard CRRA utility
    if abs(eta - 1.0) < 1e-8:
        util = np.log(c_safe)
    else:
        util = (c_safe**(1.0 - eta) - 1.0) / (1.0 - eta)
    
    # Subtract pollution disutility
    # Ensure Z doesn't cause weirdness if negative during solver exploration
    z_safe = np.maximum(z, 0.0) 
    return util - omega * z_safe

def simulate_polluting_smooth(C, x, q, params):
    """
    Simulation WITHOUT internal clipping.
    Allows K, X, Z to drift into invalid territory (penalized later).
    """
    T = params.T
    dt = params.dt
    
    K = np.zeros(T + 1)
    X = np.zeros(T + 1)
    Z = np.zeros(T + 1)
    A = np.zeros(T + 1)
    
    K[0] = params.K0
    X[0] = params.X0
    Z[0] = params.Z0
    A[0] = params.A0
    
    Y_path = np.zeros(T)
    
    for t in range(T):
        # TFP Growth
        A[t+1] = A[t] * np.exp(params.g_A * dt)
        
        # Production
        Y = production(K[t], x[t], A[t], params.alpha, params.beta_r)
        Y_path[t] = Y
        
        # Capital Motion: K_next = K + (Y - C - q) - depreciation
        # Note: q is now a cost!
        investment = Y - C[t] - q[t]
        K[t+1] = (1 - params.delta_K * dt) * K[t] + investment * dt
        
        # Resource Motion
        X[t+1] = X[t] - x[t] * dt
        
        # Pollution Motion
        emissions = params.theta * x[t]
        abated = abatement_func(q[t], params.chi, params.nu)
        
        # Net emission (cannot be lower than -natural decay in reality, 
        # but here we allow math to flow)
        net_emission = emissions - abated
        
        # Z_next = Z + (Net_Emission - Decay)
        Z[t+1] = Z[t] + (net_emission - params.phi_Z * Z[t]) * dt
        
    return K, X, Z, A, Y_path

# =========================
# 3. Objective with Terminal Value & Soft Penalties
# =========================
def polluting_objective(z_vector, params):
    T = params.T
    dt = params.dt
    beta = params.discount_factor()
    
    # Unpack Controls
    C = z_vector[:T]
    x = z_vector[T:2*T]
    q = z_vector[2*T:]
    
    # Run Simulation
    K, X, Z, A, Y = simulate_polluting_smooth(C, x, q, params)
    
    # --- UTILITY FLOW ---
    utility = 0.0
    df = 1.0
    for t in range(T):
        utility += df * u_poll(C[t], Z[t], params.eta, params.omega)
        df *= beta
        
    # --- TERMINAL VALUE (Economic + Climate) ---
    
    # 1. Economic Value (Positive)
    K_final = np.maximum(K[-1], 1e-1)
    x_final = np.maximum(x[-1] * 0.95, 1e-6) # Assume declining extraction
    Y_final = production(K_final, x_final, A[-1], params.alpha, params.beta_r)
    # Sustainable Consumption (Net of depreciation AND maintenance abatement)
    # We assume we maintain q at current ratio to keep pollution steady-ish
    q_ratio = np.clip(q[-1]/Y[-1], 0, 0.2) if Y[-1] > 0 else 0
    C_sust = np.maximum(Y_final * (1 - q_ratio) - params.delta_K * K_final, 1e-3)
    
    # 2. Climate Disutility (Negative)
    Z_final = np.maximum(Z[-1], 0.0)
    
    # Perpetuity Multiplier (~50 years horizon)
    multiplier = 0.0
    df_term = df
    for _ in range(10): # 10 periods extension
        multiplier += df_term
        df_term *= beta
        
    # Terminal Utility = Utility from Sust. Cons - Disutility from lingering Z
    term_val = multiplier * u_poll(C_sust, Z_final, params.eta, params.omega)
    
    total_welfare = utility + term_val
    
    # --- SOFT PENALTIES ---
    penalty = 0.0
    scale = 1e5
    
    # 1. Negative Stocks
    penalty += scale * np.sum(np.minimum(K, 0.0)**2)
    penalty += scale * np.sum(np.minimum(X, 0.0)**2)
    penalty += scale * np.sum(np.minimum(Z, 0.0)**2) # Pollution can't be negative
    
    # 2. Terminal Resource Non-Negativity
    if X[-1] < 0:
        penalty += scale * (X[-1]**2) * 10
        
    return -(total_welfare - penalty)

# =========================
# 4. Initial Guess (Solow + Small Abatement)
# =========================
def get_initial_guess_polluting(params):
    T = params.T
    
    # 1. Extraction: Exponential decay
    t_grid = np.arange(T) * params.dt
    x_guess = np.exp(-0.02 * t_grid)
    total_x = np.sum(x_guess) * params.dt
    x_guess = x_guess * (params.X0 / total_x) * 0.99
    
    # 2. Simulation for C and q
    K_curr = params.K0
    A_curr = params.A0
    C_guess = np.zeros(T)
    q_guess = np.zeros(T)
    
    for t in range(T):
        Y = production(K_curr, x_guess[t], A_curr, params.alpha, params.beta_r)
        
        # Guess: 70% Consumption, 2% Abatement
        C_guess[t] = 0.70 * Y 
        q_guess[t] = 0.02 * Y 
        
        # Rough update
        I = Y - C_guess[t] - q_guess[t]
        K_curr = (1 - params.delta_K*params.dt)*K_curr + I*params.dt
        A_curr *= np.exp(params.g_A * params.dt)
        
    return np.concatenate([C_guess, x_guess, q_guess])

# =========================
# 5. Execution
# =========================
if __name__ == "__main__":
    params = ParamsPolluting()
    print("Solving Enhanced Polluting Resource Model (Smooth L-BFGS-B)...")
    
    z0 = get_initial_guess_polluting(params)
    
    # Bounds: C, x, q > 0
    bounds = [(1e-3, None) for _ in range(3 * params.T)]
    
    result = minimize(
        polluting_objective,
        z0,
        args=(params,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 20000, 'ftol': 1e-9, 'disp': True}
    )
    
    print(f"\nOptimization Success: {result.success}")
    print(result.message)
    
    # Extract results
    T = params.T
    C_opt = result.x[:T]
    x_opt = result.x[T:2*T]
    q_opt = result.x[2*T:]
    
    K_opt, X_opt, Z_opt, A_opt, Y_opt = simulate_polluting_smooth(C_opt, x_opt, q_opt, params)
    
    # Diagnostics
    print("\n--- Diagnostics ---")
    print(f"Total Extraction: {np.sum(x_opt)*params.dt:.4f} / {params.X0}")
    print(f"Final Capital: {K_opt[-1]:.2f}")
    print(f"Final Pollution: {Z_opt[-1]:.2f}")
    
    c_growth = (C_opt[-1]/C_opt[0] - 1)*100
    print(f"Consumption Growth: {c_growth:.2f}%")
    
    abatement_share = np.mean(q_opt / Y_opt) * 100
    print(f"Avg Abatement Share: {abatement_share:.2f}% of Output")

    # Plot
    t = np.arange(params.T) * params.dt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1
    ax[0,0].plot(t, C_opt, label='Consumption', color='green')
    ax[0,0].set_title("Consumption")
    
    ax[0,1].plot(t, x_opt, label='Extraction', color='orange')
    ax[0,1].set_title("Resource Extraction")
    
    ax[0,2].plot(t, q_opt, label='Abatement ($)', color='red')
    ax[0,2].set_title("Abatement Spending")
    
    # Row 2
    ax[1,0].plot(np.arange(T+1)*params.dt, K_opt, color='blue')
    ax[1,0].set_title("Capital Stock")
    
    ax[1,1].plot(np.arange(T+1)*params.dt, Z_opt, color='purple')
    ax[1,1].set_title("Pollution Stock")
    
    # Abatement Ratio
    ax[1,2].plot(t, q_opt/Y_opt * 100, color='brown')
    ax[1,2].set_title("Abatement % of Output")
    
    for a in ax.flatten(): a.grid(True)
    plt.tight_layout()
    plt.show()





