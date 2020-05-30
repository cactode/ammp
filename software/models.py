import numpy as np

from objects import Conditions, Data, Prediction, MachineChar


def calc_f_t(conditions: Conditions):
    D, W, f_r, w, endmill = conditions.unpack()
    N = endmill.N
    return (2 * np.pi * f_r) / (N * w)


def calc_h_a(conditions: Conditions):
    D, W, f_r, w, endmill = conditions.unpack()
    R = endmill.r_c
    f_t = calc_f_t(conditions)
    phi_st = np.pi - np.arccos(1 - W / R)
    phi_ex = np.pi
    return -f_t * (np.cos(phi_ex) - np.cos(phi_st)) / (phi_ex - phi_st)


# exponential models, account for strange decrease in cutting forces as chip thickness increases.
def T_exp_lin(conditions: Conditions, K_TC, K_te, p):
    D, W, f_r, w, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)
    h_a = calc_h_a(conditions)
    K_tc = K_TC * h_a ** -p
    return ((D * N) * (K_te * R * np.arccos(1 - W/R) + K_tc * W * f_t)) / (2 * np.pi)


def F_exp_lin(conditions: Conditions, K_TC, K_te, K_RC, K_re, p, q):
    D, W, f_r, w, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)
    h_a = calc_h_a(conditions)
    K_tc = K_TC * h_a ** -p
    K_rc = K_RC * h_a ** -q
    Fx = - ((D*N*(K_tc*W**2*f_t + K_rc*W**(3/2)*f_t*(2*R - W)**(1/2)))/4 - (D*N*R*(2*K_tc*W*f_t - 2*K_re*W + 2*K_te*W**(1/2)
                                                                                   * (2*R - W)**(1/2) + K_rc*W**(1/2)*f_t*(2*R - W)**(1/2)))/4)/(R**2*np.pi) - (D*K_rc*N*f_t*np.arccos((R - W)/R))/(4*np.pi)
    Fy = (D*K_tc*N*f_t*np.arccos((R - W)/R))/(4*np.pi) - ((D*N*(K_rc*W**2*f_t - K_tc*W**(3/2)*f_t*(2*R - W)**(1/2)))/4 -
                                                          (D*N*R*(2*K_te*W + 2*K_rc*W*f_t + 2*K_re*W**(1/2)*(2*R - W)**(1/2) - K_tc*W**(1/2)*f_t*(2*R - W)**(1/2)))/4)/(R**2*np.pi)
    return np.array([Fx, Fy])

# linear models, easier to fit and more simple


def T_lin(conditions, K_tc, K_te):
    D, W, f_r, w, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)
    return ((D * N) * (K_te * R * np.arccos(1 - W/R) + K_tc * W * f_t)) / (2 * np.pi)


def F_lin(conditions, K_tc, K_te, K_rc, K_re):
    D, W, f_r, w, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)
    Fx = - ((D*N*(K_tc*W**2*f_t + K_rc*W**(3/2)*f_t*(2*R - W)**(1/2)))/4 - (D*N*R*(2*K_tc*W*f_t - 2*K_re*W + 2*K_te*W**(1/2)
                                                                                   * (2*R - W)**(1/2) + K_rc*W**(1/2)*f_t*(2*R - W)**(1/2)))/4)/(R**2*np.pi) - (D*K_rc*N*f_t*np.arccos((R - W)/R))/(4*np.pi)
    Fy = (D*K_tc*N*f_t*np.arccos((R - W)/R))/(4*np.pi) - ((D*N*(K_rc*W**2*f_t - K_tc*W**(3/2)*f_t*(2*R - W)**(1/2)))/4 -
                                                          (D*N*R*(2*K_te*W + 2*K_rc*W*f_t + 2*K_re*W**(1/2)*(2*R - W)**(1/2) - K_tc*W**(1/2)*f_t*(2*R - W)**(1/2)))/4)/(R**2*np.pi)
    return np.array([Fx, Fy])

# coefficient calculators for linear regression


def T_x_vector(conditions: Conditions):
    """
    Outputs vector that corresponds to coefficients in order:
        K_tc, K_te
    """
    D, W, f_r, w, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)

    return[(D * N * W * f_t) / (2 * np.pi), (D * N * R * np.arccos(1 - (W / R))) / (2 * np.pi)]


def Fy_x_vector(conditions: Conditions):
    D, W, f_r, w, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)
    return [(D*N*f_t*(W**(3/2)*(2*R - W)**(1/2) + R**2*np.arccos((R - W)/R) - R*W**(1/2)*(2*R - W)**(1/2)))/(4*R**2*np.pi), (D*N*W)/(2*R*np.pi), (D*N*W*f_t*(2*R - W))/(4*R**2*np.pi), (D*N*W**(1/2)*(2*R - W)**(1/2))/(2*R*np.pi)]

def deflection_load(D_a, prediction: Prediction, E_e=650e9):
    """
    Calculates ratio of tool deflection to allowed deflection.
    Uses FEA model described in doi:10.1016/j.ijmachtools.2005.09.009.

    Args:
        D_a: Allowed deflection (m)
        prediction: A prediction object 
        E_e: Modulus of elasticity (default is for Carbide) (Pa)

    Returns:
        Ratio of tool deflection to allowed deflection
    """
    D, W, f_r, w, endmill, T, F = prediction.unpack()
    F = np.linalg.norm(F)
    N, r_c, r_s, l_c, l_s = endmill.unpack()

    # prepare variables, this must be done since the FEA model has arbitrary constants defined for certain units
    d_1 = r_c * 2 * 1e3  # convert to diameter in mm
    d_2 = r_s * 2 * 1e3
    l_1 = l_c * 1e3
    l_2 = (l_s + l_c) * 1e3
    E = E_e * 1e-6  # convert to MPa

    # set constants
    C = None
    G = None
    if N == 4:
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
    D_m = C * (F / E) * ((l_1 ** 3 / d_1 ** 4) +
                         ((l_2 ** 3 - l_1 ** 3) / d_2 ** 4)) ** G * 1e-3
    return D_m / D_a


def failure_prob_milling(prediction: Prediction, roughing=False, m=4, o=1500e6, a_c=0.8):
    """
    Calculates failure probability according to Weibull distribution. Method adapted from https://doi.org/10.1016/S0007-8506(07)62072-1

    Args:
        prediction: A prediction object
        roughing: Whether or not the cutter has roughing serrations (true/false)
        m: Weibull parameter for carbide tooling (dimensionless)
        o: Weibull parameter for carbide tooling (N)
        a_c: Compensated cutter diameter due to helical flutes (dimensionless)

    Returns:
        Probability of failure
    """
    D, W, f_r, w, endmill, T, F = prediction.unpack()
    F = np.linalg.norm(F)
    N, r_c, r_s, l_c, l_s = endmill.unpack()      # establish base variables

    r_ceq = r_c * a_c  # equivalent cutter radius
    I_ceq = np.pi / 4 * r_ceq ** 4  # equivalent cutter 2nd inertia
    I_s = np.pi / 4 * r_s ** 4  # shank 2nd inertia
    J_ceq = np.pi * r_ceq ** 4 / 2  # equivalent cutter polar inertia

    def stress_tensile(r, l, I): return (F * r * (l - D / 2)) / I

    # finding surface stress at cutter-shank interface at the depth of the cutter edge
    stress_c = stress_tensile(r_c, l_c, I_ceq)
    # finding surface stress at shaft-collet interface
    stress_s = stress_tensile(r_s, l_c + l_s, I_s)
    # adjusting torsion to compensate for shear-tensile difference
    stress_t = 1 / 0.577 * T * r_c / J_ceq

    if roughing:
        stress_c *= 2  # accounting for stress factors

    stress_peak = max(stress_c, stress_s, stress_t)

    failure_prob = 1 - np.exp(-(stress_peak / o) ** m)
    return failure_prob


def motor_torque_load(prediction: Prediction, machinechar: MachineChar):
    """
    Calculates ratio of current motor torque to max motor torque.

    Args:
        prediction: A prediction object.
        machinechar: A machinechar object.
    Returns:
        Ratio of current motor torque to max achievable torque
    """
    _, _, _, w, _, T, _ = prediction.unpack()
    r_e, K_T, R_w, V_max, I_max, T_nom, _ = machinechar.unpack()
    w_m = w * r_e
    T_m = (T + T_nom) / r_e
    # store K_V for convenience
    K_V = 1 / K_T
    # max torque is either determined by max current that can be supplied or max current achievable given winding resistance and whatnot
    return T_m / min(K_T * I_max, (V_max - K_V * w_m) / R_w)


def motor_speed_load(prediction: Prediction, machinechar: MachineChar):
    """
    Calculates ratio of motor speed to maximum motor speed
    Args:
        prediction: A prediction object
        machinechar: A machinechar object

    Returns:
        Ratio of current motor speed to max achievable speed
    """
    _, _, _, w, _, T, _ = prediction.unpack()
    r_e, K_T, R_w, V_max, I_max, T_nom, _ = machinechar.unpack()
    w_m = w * r_e
    T_m = (T + T_nom) / r_e
    # store K_V for convenience
    K_V = 1 / K_T
    # max speed is affected by winding resistive voltage drop along with backemf
    return w_m / (K_V * (V_max - T_m * R_w / K_T))


def optimality(conditions: Conditions):
    """
    A measure of productivity for the cutting process. Specifically, MMR

    Args:
        conditions: A conditions object.

    Returns:
        The material removal rate (m^3 / s)
    """
    D, W, f_r, _, _ = conditions.unpack()
    # returns MMR in units of m^3 / s
    return D * W * f_r
