import numpy as np

def calc_h_a(R, W, f_t):
    phi_st = np.pi - np.arccos(1 - W / R)
    phi_ex = np.pi
    return -f_t * (np.cos(phi_ex) - np.cos(phi_st)) / (phi_ex - phi_st)

def calc_f_t(N, f_r, w):
    return (2 * np.pi * f_r) / (N * w)

# exponential models, account for strange decrease in cutting forces as chip thickness increases.
def T_exp_lin(D, N, R, W, f_r, w, K_TC, K_te, p):
    f_t = calc_f_t(N, f_r, w)
    h_a = calc_h_a(R, W, f_t)
    K_tc = K_TC * h_a ** -p
    return ((D * N) * (K_te * R * np.arccos(1 - W/R) + K_tc * W * f_t)) / (2 * np.pi)

def F_exp_lin(D, N, R, W, f_r, w, K_TC, K_te, K_RC, K_re, p, q):
    f_t = calc_f_t(N, f_r, w)
    h_a = calc_h_a(R, W, f_t)
    K_tc = K_TC * h_a ** -p
    K_rc = K_RC * h_a ** -q
    Fx = - ((D*N*(K_tc*W**2*f_t + K_rc*W**(3/2)*f_t*(2*R - W)**(1/2)))/4 - (D*N*R*(2*K_tc*W*f_t - 2*K_re*W + 2*K_te*W**(1/2)*(2*R - W)**(1/2) + K_rc*W**(1/2)*f_t*(2*R - W)**(1/2)))/4)/(R**2*np.pi) - (D*K_rc*N*f_t*np.arccos((R - W)/R))/(4*np.pi)
    Fy = (D*K_tc*N*f_t*np.arccos((R - W)/R))/(4*np.pi) - ((D*N*(K_rc*W**2*f_t - K_tc*W**(3/2)*f_t*(2*R - W)**(1/2)))/4 - (D*N*R*(2*K_te*W + 2*K_rc*W*f_t + 2*K_re*W**(1/2)*(2*R - W)**(1/2) - K_tc*W**(1/2)*f_t*(2*R - W)**(1/2)))/4)/(R**2*np.pi)
    return np.array([Fx, Fy])

# linear models, easier to fit and more simple

def T_lin(D, N, R, W, f_r, w, K_tc, K_te):
    f_t = calc_f_t(N, f_r, w)
    return ((D * N) * (K_te * R * np.arccos(1 - W/R) + K_tc * W * f_t)) / (2 * np.pi)

def F_lin(D, N, R, W, f_r, w, K_tc, K_te, K_rc, K_re):
    f_t = calc_f_t(N, f_r, w)
    Fx = - ((D*N*(K_tc*W**2*f_t + K_rc*W**(3/2)*f_t*(2*R - W)**(1/2)))/4 - (D*N*R*(2*K_tc*W*f_t - 2*K_re*W + 2*K_te*W**(1/2)*(2*R - W)**(1/2) + K_rc*W**(1/2)*f_t*(2*R - W)**(1/2)))/4)/(R**2*np.pi) - (D*K_rc*N*f_t*np.arccos((R - W)/R))/(4*np.pi)
    Fy = (D*K_tc*N*f_t*np.arccos((R - W)/R))/(4*np.pi) - ((D*N*(K_rc*W**2*f_t - K_tc*W**(3/2)*f_t*(2*R - W)**(1/2)))/4 - (D*N*R*(2*K_te*W + 2*K_rc*W*f_t + 2*K_re*W**(1/2)*(2*R - W)**(1/2) - K_tc*W**(1/2)*f_t*(2*R - W)**(1/2)))/4)/(R**2*np.pi)
    return np.array([Fx, Fy])
    
# coefficient calculators for linear regression

def T_x_vector(D, N, R, W, f_r, w):
    """
    Outputs vector that corresponds to coefficients in order:
        K_tc, K_te
    """
    f_t = calc_f_t(N, f_r, w)
    K_tc_C = (D * N * W * f_t) / (2 * np.pi)
    K_te_C = (D * N * R * np.arccos(1 - (W / R))) / (2 * np.pi)
    
    return [K_tc_C, K_te_C]

def Fy_x_vector(D, N, R, W, f_r, w):
    f_t = calc_f_t(N, f_r, w)
    return [(D*N*f_t*(W**(3/2)*(2*R - W)**(1/2) + R**2*np.arccos((R - W)/R) - R*W**(1/2)*(2*R - W)**(1/2)))/(4*R**2*np.pi), (D*N*W)/(2*R*np.pi), (D*N*W*f_t*(2*R - W))/(4*R**2*np.pi), (D*N*W**(1/2)*(2*R - W)**(1/2))/(2*R*np.pi)]



def deflection_load(D_a, F, endmill, E_e = 650e9):
    """
    Calculates ratio of tool deflection to allowed deflection.
    Uses FEA model described in doi:10.1016/j.ijmachtools.2005.09.009.
    Args:
        D_a: Allowed deflection (m)
        F: Radial cutting force (N)
        endmill: Endmill
        E_e: Modulus of elasticity (default is for Carbide) (Pa)
    Returns:
        Ratio of tool deflection to allowed deflection
    """
    r_c = endmill.r_c
    r_s = endmill.r_s
    l_c = endmill.l_c
    l_s = endmill.l_s
    N = endmill.N
    # prepare variables, this must be done since the FEA model has arbitrary constants defined for certain units
    d_1 = r_c * 2 * 1e3 # convert to diameter in mm
    d_2 = r_s * 2 * 1e3
    l_1 = l_c * 1e3
    l_2 = (l_s + l_c) * 1e3
    E = E_e * 1e-6 # convert to MPa
    
    # set constants
    C = None
    G = None
    if   N == 4:
        C = 9.05
        G = 0.950
    elif N == 3:
        C = 8.30
        G = 0.965
    elif N == 2:
        C = 7.93
        G = 0.974
    else:
        raise ValueError("Flute count must be between 2-4")
    
    # calculate tool bending
    D_m = C * (F / E) * ((l_1 ** 3 / d_1 ** 4) + ((l_2 ** 3 - l_1 ** 3) / d_2 ** 4)) ** G * 1e-3
    return D_m / D_a

def failure_prob_milling(F, T, endmill, D, roughing = False, m = 4, o = 1500e6, a_c = 0.8):
    """
    Calculates failure probability according to Weibull distribution. Method adapted from https://doi.org/10.1016/S0007-8506(07)62072-1
    Args:
        F: Radial cutting force (N)
        T: Cutting torque (N*m)
        endmill: Endmill
        l_doc: Axial DOC (m)
        roughing: Whether or not the cutter has roughing serrations (true/false)
        m: Weibull parameter for carbide tooling (dimensionless)
        o: Weibull parameter for carbide tooling (N)
        a_c: Compensated cutter diameter due to helical flutes (dimensionless)
    Returns:
        Probability of failure
    """
    # establish base variables
    r_c = endmill.r_c
    r_s = endmill.r_s
    l_c = endmill.l_c
    l_s = endmill.l_s
    r_ceq = r_c * a_c # equivalent cutter radius
    I_ceq = np.pi / 4 * r_ceq ** 4 # equivalent cutter 2nd inertia
    I_s = np.pi / 4 * r_s ** 4 # shank 2nd inertia
    J_ceq = np.pi * r_ceq ** 4 / 2 # equivalent cutter polar inertia
    
    stress_tensile = lambda r, l, I: (F * r * (l - D / 2)) / I
    
    stress_c = stress_tensile(r_c, l_c, I_ceq) # finding surface stress at cutter-shank interface at the depth of the cutter edge
    stress_s = stress_tensile(r_s, l_c + l_s, I_s) # finding surface stress at shaft-collet interface
    stress_t = 1 / 0.577 * T * r_c / J_ceq # adjusting torsion to compensate for shear-tensile difference
    
    if roughing: stress_c *= 2 # accounting for stress factors
    
    stress_peak = max(stress_c, stress_s, stress_t)
    
    failure_prob = 1 - np.exp(-(stress_peak / o) ** m)
    return failure_prob


def motor_torque_load(T_m, w, K_T, R_w, V_max, I_max):
    """
    Calculates ratio of current motor torque to max motor torque.
    Args:
        T_m: Motor torque (N*m)
        w: Motor velocity (rad/s)
        K_T: Motor torque constant (N*m/A)
        R_w: Motor winding resistance (ohm)
        V_max: Max input voltage (V)
        I_max: Max input current (A)
    Returns:
        Ratio of current motor torque to max achievable torque
    """
    # store K_V for convenience
    K_V = 1 / K_T
    # max torque is either determined by max current that can be supplied or max current achievable given winding resistance and whatnot
    return T_m / min(K_T * I_max, (V_max - K_V * w) / R_w)

def motor_speed_load(T_m, w, K_T, R_w, V_max):
    """
    Calculates ratio of 
        Args:
        T_m: Motor torque (N*m)
        w: Motor velocity (rad/s)
        K_T: Motor torque constant (N*m/A)
        R_w: Motor winding resistance (ohm)
        V_max: Max input voltage (V)
    Returns:
        Ratio of current motor speed to max achievable speed
    """
    # store K_V for convenience
    K_V = 1 / K_T
    # max speed is affected by winding resistive voltage drop along with backemf
    return w / (K_V * (V_max - T_m * R_w / K_T))

def optimality(D, W, f_r):
    # returns MMR in units of m^3 / s
    return D * W * f_r
