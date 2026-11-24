import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =========================
# 1. Parameters & Setup
# =========================
@dataclass
class ParamsSDH:
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

    def discount_factor(self):
        return np.exp(-self.rho * self.dt)

# =========================
# 2. Smooth Simulation (No Clipping)
# =========================
def production(K, x, A, alpha, beta_r):
    # Safety floor for logs/powers, but no logic changes
    K_safe = np.maximum(K, 1e-6)
    x_safe = np.maximum(x, 1e-6)
    return A * (K_safe**alpha) * (x_safe**beta_r)

def u_crra(c, eta):
    c_safe = np.maximum(c, 1e-6)
    if abs(eta - 1.0) < 1e-8:
        return np.log(c_safe)
    else:
        return (c_safe**(1.0 - eta) - 1.0) / (1.0 - eta)

def simulate_sdh_smooth(C, x, params):
    """
    Simulation WITHOUT internal clipping/min/max logic.
    Allows K and X to go negative theoretically (penalized later).
    This keeps the gradients smooth for the solver.
    """
    T = params.T
    dt = params.dt
    
    K = np.zeros(T + 1)
    X = np.zeros(T + 1)
    A = np.zeros(T + 1)
    
    K[0] = params.K0
    X[0] = params.X0
    A[0] = params.A0
    
    # We construct output arrays for analysis
    Y_path = np.zeros(T)
    
    for t in range(T):
        # Exogenous TFP
        A[t+1] = A[t] * np.exp(params.g_A * dt)
        
        # Production (Standard Cobb-Douglas)
        Y = production(K[t], x[t], A[t], params.alpha, params.beta_r)
        Y_path[t] = Y
        
        # Capital Motion: K_next = K + I - depreciation
        # Investment I = Y - C
        # Note: We do NOT force C <= Y here. We let K go negative if C is too high.
        investment = Y - C[t]
        K[t+1] = (1 - params.delta_K * dt) * K[t] + investment * dt
        
        # Resource Motion
        X[t+1] = X[t] - x[t] * dt
        
    return K, X, A, Y_path

# =========================
# 3. Objective with Penalties & Terminal Value
# =========================
def sdh_objective(z, params):
    T = params.T
    dt = params.dt
    beta = params.discount_factor()
    
    # Unpack Controls
    C = z[:T]
    x = z[T:]
    
    # Run Simulation
    K, X, A, Y = simulate_sdh_smooth(C, x, params)
    
    # --- UTILITY ---
    utility = 0.0
    df = 1.0
    for t in range(T):
        utility += df * u_crra(C[t], params.eta)
        df *= beta
        
    # --- TERMINAL VALUE (Avoids Consumption Collapse) ---
    # Value the final capital stock based on a sustainable consumption proxy
    K_final = np.maximum(K[-1], 1e-1)
    x_final = np.maximum(x[-1] * 0.95, 1e-6) # Assume declining extraction
    
    # Sustainable Output capacity at T
    Y_final = production(K_final, x_final, A[-1], params.alpha, params.beta_r)
    # Sustainable Consumption (Net of depreciation)
    C_sust = np.maximum(Y_final - params.delta_K * K_final, 1e-3)
    
    # Value as a perpetuity (approx 50 years horizon)
    multiplier = 0.0
    df_term = df
    for _ in range(10): # 10 periods (50 years) extension
        multiplier += df_term
        df_term *= beta
        
    term_val = multiplier * u_crra(C_sust, params.eta)
    
    total_welfare = utility + term_val
    
    # --- PENALTIES (Soft Constraints) ---
    penalty = 0.0
    scale = 1e5 # Strong enough to enforce, smooth enough to differentiate
    
    # 1. Negative Capital Penalty (Don't borrow from Mars)
    # If K < 0, punish quadratically
    neg_K = np.minimum(K, 0.0)
    penalty += scale * np.sum(neg_K**2)
    
    # 2. Negative Resource Penalty (Don't extract what you don't have)
    # If X < 0, punish
    neg_X = np.minimum(X, 0.0)
    penalty += scale * np.sum(neg_X**2)
    
    # 3. Terminal Resource Constraint (Ensure we don't end negative)
    if X[-1] < 0:
        penalty += scale * (X[-1]**2) * 10
        
    return -(total_welfare - penalty)

# =========================
# 4. Solow-Style Initial Guess
# =========================
def get_initial_guess(params):
    T = params.T
    
    # Guess 1: Resource extraction declines exponentially (Hotelling-ish)
    # Total sum should be close to X0
    decay = 0.02 # 2% decay per year
    t_grid = np.arange(T) * params.dt
    x_guess = np.exp(-decay * t_grid)
    # Scale to match X0 exactly
    total_x = np.sum(x_guess) * params.dt
    x_guess = x_guess * (params.X0 / total_x) * 0.99 # Use 99% of stock
    
    # Guess 2: Consumption is 75% of output (Standard macro)
    # We run a quick simulation to estimate Y
    K_sim = np.zeros(T)
    K_curr = params.K0
    A_curr = params.A0
    C_guess = np.zeros(T)
    
    for t in range(T):
        Y = production(K_curr, x_guess[t], A_curr, params.alpha, params.beta_r)
        C_guess[t] = 0.75 * Y # 75% Consumption rule
        
        # Update K roughly
        I = Y - C_guess[t]
        K_curr = (1 - params.delta_K*params.dt)*K_curr + I*params.dt
        A_curr *= np.exp(params.g_A * params.dt)
        
    return np.concatenate([C_guess, x_guess])

# =========================
# 5. Execution
# =========================
if __name__ == "__main__":
    params = ParamsSDH()
    print("Solving SDH Model with Smooth Constraints...")
    
    z0 = get_initial_guess(params)
    
    # Bounds: C > 0, x > 0. No upper bound needed (penalties handle K/X limits)
    bounds = [(1e-3, None) for _ in range(2 * params.T)]
    
    result = minimize(
        sdh_objective,
        z0,
        args=(params,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-9, 'disp': True}
    )
    
    print(f"\nOptimization Success: {result.success}")
    print(result.message)
    
    # Extract results
    C_opt = result.x[:params.T]
    x_opt = result.x[params.T:]
    K_opt, X_opt, A_opt, Y_opt = simulate_sdh_smooth(C_opt, x_opt, params)
    
    # Diagnostics
    print("\n--- Diagnostics ---")
    print(f"Total Extraction: {np.sum(x_opt)*params.dt:.4f} / {params.X0}")
    print(f"Final Capital: {K_opt[-1]:.2f}")
    
    c_growth = (C_opt[-1]/C_opt[0] - 1)*100
    print(f"Consumption Growth: {c_growth:.2f}%")
    
    # Plot
    t = np.arange(params.T) * params.dt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    
    ax[0].plot(t, C_opt, label='Consumption')
    ax[0].set_title("Consumption Path")
    ax[0].grid(True)
    
    ax[1].plot(t, x_opt, color='orange', label='Extraction')
    ax[1].set_title("Resource Extraction")
    ax[1].grid(True)
    
    ax[2].plot(np.arange(params.T+1)*params.dt, K_opt, color='green', label='Capital')
    ax[2].set_title("Capital Accumulation")
    ax[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    
    
