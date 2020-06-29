import numpy as np

from objects import Conditions, Data, Prediction, MachineChar


def calc_f_t(conditions: Conditions):
    _, _, f_r, w, endmill = conditions.unpack()
    N = endmill.N
    return (2 * np.pi * f_r) / (N * w)


def calc_h_a(conditions: Conditions):
    _, W, _, _, endmill = conditions.unpack()
    R = endmill.r_c
    f_t = calc_f_t(conditions)
    phi_st = np.pi - np.arccos(1 - W / R)
    phi_ex = np.pi
    return -f_t * (np.cos(phi_ex) - np.cos(phi_st)) / (phi_ex - phi_st)


# exponential models, account for strange decrease in cutting forces as chip thickness increases.
def T_exp_lin(conditions: Conditions, K_TC, K_te, p):
    D, W, _, _, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)
    h_a = calc_h_a(conditions)
    K_tc = K_TC * h_a ** -p
    return ((D * N) * (K_te * R * np.arccos(1 - W/R) + K_tc * W * f_t)) / (2 * np.pi)


def F_exp_lin(conditions: Conditions, K_TC, K_te, K_RC, K_re, p, q):
    D, W, _, _, endmill = conditions.unpack()
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
def T_lin(conditions, K_tc, K_te, K_rc, K_re):
    D, W, _, _, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)
    return ((D * N) * (K_te * R * np.arccos(1 - W/R) + K_tc * W * f_t)) / (2 * np.pi)


def F_lin(conditions, K_tc, K_te, K_rc, K_re):
    D, W, _, _, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)
    Fx = - ((D*N*(K_tc*W**2*f_t + K_rc*W**(3/2)*f_t*(2*R - W)**(1/2)))/4 - (D*N*R*(2*K_tc*W*f_t - 2*K_re*W + 2*K_te*W**(1/2)
                                                                                   * (2*R - W)**(1/2) + K_rc*W**(1/2)*f_t*(2*R - W)**(1/2)))/4)/(R**2*np.pi) - (D*K_rc*N*f_t*np.arccos((R - W)/R))/(4*np.pi)
    Fy = (D*K_tc*N*f_t*np.arccos((R - W)/R))/(4*np.pi) - ((D*N*(K_rc*W**2*f_t - K_tc*W**(3/2)*f_t*(2*R - W)**(1/2)))/4 -
                                                          (D*N*R*(2*K_te*W + 2*K_rc*W*f_t + 2*K_re*W**(1/2)*(2*R - W)**(1/2) - K_tc*W**(1/2)*f_t*(2*R - W)**(1/2)))/4)/(R**2*np.pi)
    return np.array([Fx, Fy])

# linear models that account for spindle speed
def T_lin_full(conditions, K_tc, K_tc0, K_te, K_rc, K_rc0, K_re):
    """
    Linear model that also attempts to account for variations in cutting force from spindle speeds. Assumes a linear (w/ intercept) relation between the two.
    Args:
        conditions: Conditions for cut
        K_tc: Cutting force coefficient, in units of (Pa / (rad/s)) / (m)
        K_tc0: Cutting force coefficient, in units of Pa
        K_te: Edge force coefficient, in units of (Pa)
    """
    D, W, f_r, w, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    return (D*N*R*(K_te*np.arccos(1 - W/R) + (2*np.pi*K_tc0*W*f_r + 2*np.pi*K_tc*R*W*f_r*w)/(N*R*w)))/(2*np.pi)

def F_lin_full(conditions, K_tc, K_tc0, K_te, K_rc, K_rc0, K_re):
    """
    Linear model that also attempts to account for variations in cutting force from spindle speeds. Assumes a linear (w/ intercept) relation between the two.
    """
    D, W, f_r, w, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    return  [- ((D*K_re*N*R*W)/2 + (D*K_rc*R**3*f_r*np.pi*np.arccos((R - W)/R))/2 - (D*K_te*N*R*W**(1/2)*(2*R - W)**(1/2))/2 - (D*K_tc*R*W*f_r*np.pi*(2*R - W))/2 - (D*K_rc*R**2*W**(1/2)*f_r*np.pi*(2*R - W)**(1/2))/2 + (D*K_rc*R*W**(3/2)*f_r*np.pi*(2*R - W)**(1/2))/2)/(R**2*np.pi) - ((D*K_rc0*R**2*f_r*np.pi*np.arccos((R - W)/R))/2 + (D*K_rc0*W**(3/2)*f_r*np.pi*(2*R - W)**(1/2))/2 - (D*K_tc0*W*f_r*np.pi*(2*R - W))/2 - (D*K_rc0*R*W**(1/2)*f_r*np.pi*(2*R - W)**(1/2))/2)/(R**2*w*np.pi),
               ((D*K_te*N*R*W)/2 + (D*K_tc*R**3*f_r*np.pi*np.arccos((R - W)/R))/2 + (D*K_re*N*R*W**(1/2)*(2*R - W)**(1/2))/2 + (D*K_rc*R*W*f_r*np.pi*(2*R - W))/2 - (D*K_tc*R**2*W**(1/2)*f_r*np.pi*(2*R - W)**(1/2))/2 + (D*K_tc*R*W**(3/2)*f_r*np.pi*(2*R - W)**(1/2))/2)/(R**2*np.pi) + ((D*K_tc0*R**2*f_r*np.pi*np.arccos((R - W)/R))/2 + (D*K_tc0*W**(3/2)*f_r*np.pi*(2*R - W)**(1/2))/2 + (D*K_rc0*W*f_r*np.pi*(2*R - W))/2 - (D*K_tc0*R*W**(1/2)*f_r*np.pi*(2*R - W)**(1/2))/2)/(R**2*w*np.pi)]

# coefficient calculators for linear regression
def T_x_vector(conditions: Conditions):
    """
    Outputs vector that corresponds to coefficients in order:
        K_tc, K_te
    """
    D, W, _, _, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)

    return[(D * N * W * f_t) / (2 * np.pi), (D * N * R * np.arccos(1 - (W / R))) / (2 * np.pi)]

def T_x_vector_padded(conditions: Conditions):
    """
    Outputs vector that corresponds to coefficients in order:
        K_tc, K_te, K_rc, K_re
    Note that last two coefficents are 0 no matter what.
    """
    D, W, _, _, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)

    return[(D * N * W * f_t) / (2 * np.pi), (D * N * R * np.arccos(1 - (W / R))) / (2 * np.pi), 0, 0]

def Fy_x_vector(conditions: Conditions):
    D, W, _, _, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    f_t = calc_f_t(conditions)
    return [(D*N*f_t*(W**(3/2)*(2*R - W)**(1/2) + R**2*np.arccos((R - W)/R) - R*W**(1/2)*(2*R - W)**(1/2)))/(4*R**2*np.pi), (D*N*W)/(2*R*np.pi), (D*N*W*f_t*(2*R - W))/(4*R**2*np.pi), (D*N*W**(1/2)*(2*R - W)**(1/2))/(2*R*np.pi)]

def T_x_vector_full(conditions: Conditions):
    D, W, f_r, w, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    return [ D*R*W*f_r, (D*W*f_r)/w, (D*N*R*np.arccos(1 - W/R))/(2*np.pi), 0, 0, 0]

def Fy_x_vector_full(conditions: Conditions):
    D, W, f_r, w, endmill = conditions.unpack()
    N, R, _, _, _ = endmill.unpack()
    return [ (D*f_r*(W**(3/2)*(2*R - W)**(1/2) + R**2*np.arccos((R - W)/R) - R*W**(1/2)*(2*R - W)**(1/2)))/(2*R), (D*f_r*(W**(3/2)*(2*R - W)**(1/2) + R**2*np.arccos((R - W)/R) - R*W**(1/2)*(2*R - W)**(1/2)))/(2*R**2*w), (D*N*W)/(2*R*np.pi), (D*W*f_r*(2*R - W))/(2*R), (D*W*f_r*(2*R - W))/(2*R**2*w), (D*N*W**(1/2)*(2*R - W)**(1/2))/(2*R*np.pi)]

def deflection_load(D_a, prediction: Prediction, machinechar : MachineChar, E_e=650e9):
    """
    Calculates ratio of tool deflection to allowed deflection.
    Uses FEA model described in doi:10.1016/j.ijmachtools.2005.09.009.
    Also includes machine deflection if it is defined.

    Args:
        D_a: Allowed deflection (m)
        prediction: A prediction object 
        E_e: Modulus of elasticity (default is for Carbide) (Pa)

    Returns:
        Ratio of tool deflection to allowed deflection
    """
    _, _, _, _, endmill, _, F = prediction.unpack()
    K_machine = machinechar.K_machine
    Fy = F[1]
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
    D_t = C * (Fy / E) * ((l_1 ** 3 / d_1 ** 4) +
                         ((l_2 ** 3 - l_1 ** 3) / d_2 ** 4)) ** G * 1e-3

    # calculate machine deflection
    D_m = Fy / K_machine

    return (D_t + D_m) / D_a


def deflection_load_simple(prediction: Prediction, machinechar : MachineChar, E_e = 600e9, a_c = 0.8):
    """
    Calculates ratio of deflection to allowable deflection.
    Uses simple approximation that assumes endmill cutter can be approx. as 0.8 of its diameter.

    Args:
        D_a: Allowed deflection (m)
        prediction: Prediction of force
        machinechar: Characterization of machine
        E_e: Modulus of elasticity for endmill
        a_c: Diameter reduction of cutter for cantilever beam model of endmill.

    Returns:
        Ratio of tool deflection to allowed deflection along y axis (since this is the only axis that matters for milling a face)
    """
    D, _, _, _, endmill, _, F = prediction.unpack()
    _, _, _, _, _, _, _, K_machine, D_a = machinechar.unpack()
    F_y = F[1]
    _, r_c, r_s, l_c, l_s = endmill.unpack()

    # calculate basic params
    D_m = F_y / K_machine
    r_ceq = r_c * a_c  # equivalent cutter radius
    I_ceq = np.pi / 4 * r_ceq ** 4  # equivalent cutter 2nd inertia
    I_s = np.pi / 4 * r_s ** 4  # shank 2nd inertia

    M_s = F_y * (l_c - D / 2) # find moment at end of shank
    D_s = (M_s * l_s ** 2) / (2 * E_e * I_s) + (F_y * l_s ** 3) / (3 * E_e * I_s) # deflection of shank from moment and force
    theta_s = (M_s * l_s) / (E_e * I_s) # slope at end of shank
    D_c = (F_y * (l_c - D / 2) ** 2 * (3 * l_c - (l_c - D / 2))) / (6 * E_e * I_ceq) + D_s + theta_s * l_c # deflection at cutter tip from everything

    return (D_m + D_c) / D_a

def failure_prob_milling(prediction: Prediction, m=4, o=1500e6, a_c=0.8):
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
    D, _, _, _, endmill, T, F = prediction.unpack()
    F = np.linalg.norm(F)
    _, r_c, r_s, l_c, l_s = endmill.unpack()      # establish base variables

    r_ceq = r_c * a_c  # equivalent cutter radius
    I_ceq = np.pi / 4 * r_ceq ** 4  # equivalent cutter 2nd inertia
    I_s = np.pi / 4 * r_s ** 4  # shank 2nd inertia
    J_ceq = np.pi * r_ceq ** 4 / 2  # equivalent cutter polar inertia

    # reusable expression to get peak stress in cutter
    def stress_tensile(r, l, I): return (F * r * (l - D / 2)) / I

    # finding surface stress at cutter-shank interface at the depth of the cutter edge
    stress_c = stress_tensile(r_c, l_c, I_ceq)
    # finding surface stress at shaft-collet interface
    stress_s = stress_tensile(r_s, l_c + l_s, I_s)
    # adjusting torsion to compensate for shear-tensile difference
    stress_t = 1 / 0.577 * T * r_c / J_ceq

    # unknown if von mises stress criterion makes any sense for weibull analysis, taking max of all stresses instead
    stress_peak = max(stress_c, stress_s, stress_t)

    # cdf for weibull distribution
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
    r_e, K_T, R_w, V_max, I_max, T_nom, _, _, _ = machinechar.unpack()
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
    r_e, K_T, R_w, V_max, _, T_nom, _, _, _ = machinechar.unpack()
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
