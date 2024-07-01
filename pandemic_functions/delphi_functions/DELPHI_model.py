import numpy as np
from pandemic_functions.pandemic_params import (
    p_d, p_h, p_v,
    IncubeD,
    RecoverID,
    RecoverHD,
    DetectD,
    VentilatedD,
)

def model_covid(
        t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal,
        N, policy_scenario_gammas, policy_startT, maxT:int
    ):
    """
    SEIR based model with 16 distinct states, taking into account undetected, deaths, hospitalized
    and recovered, and using an ArcTan government response curve, corrected with a Gaussian jump in
    case of a resurgence in cases
    :param t: time step
    :param x: set of all the states in the model (here, 16 of them)
    :param alpha: Infection rate
    :param days: Median day of action (used in the arctan governmental response)
    :param r_s: Median rate of action (used in the arctan governmental response)
    :param r_dth: Rate of death
    :param p_dth: Initial mortality percentage
    :param r_dthdecay: Rate of decay of mortality percentage
    :param k1: Internal parameter 1 (used for initial conditions)
    :param k2: Internal parameter 2 (used for initial conditions)
    :param jump: Amplitude of the Gaussian jump modeling the resurgence in cases
    :param t_jump: Time where the Gaussian jump will reach its maximum value
    :param std_normal: Standard Deviation of the Gaussian jump (~ time span of resurgence in cases)
    :param N: total population of the region
    :param policy_scenario_gammas: dictionary of time period => effective gamma
    :param policy_startT: time step of beginning of first policy
    :param maxT: maximum time step
    :return: predictions for all 16 states, which are the following
    [0 S, 1 E, 2 I, 3 UR, 4 DHR, 5 DQR, 6 UD, 7 DHD, 8 DQD, 9 R, 10 D, 11 TH,
    12 DVR,13 DVD, 14 DD, 15 DT]
    """
    r_i = np.log(2) / IncubeD  # Rate of infection leaving incubation phase
    r_d = np.log(2) / DetectD  # Rate of detection
    r_ri = np.log(2) / RecoverID  # Rate of recovery not under infection
    r_rh = np.log(2) / RecoverHD  # Rate of recovery under hospitalization
    r_rv = np.log(2) / VentilatedD  # Rate of recovery under ventilation
    gamma_t = (
            (2 / np.pi) * np.arctan(-(t - days) / 20 * r_s) + 1 +
            jump * np.exp(-(t - t_jump)**2 /(2 * std_normal ** 2))
    )
    p_dth_mod = (2 / np.pi) * (p_dth - 0.01) * (np.arctan(- t / 20 * r_dthdecay) + np.pi / 2) + 0.01
    if policy_scenario_gammas is not None and policy_startT is not None:
        if t >= policy_startT and t<maxT:
            for window, gamma in policy_scenario_gammas.items():
                if t >= window[0] and t<window[1]:
                    gamma_t = gamma
                    break

    assert len(x) == 16, f"Too many input variables, got {len(x)}, expected 16"
    S, E, I, AR, DHR, DQR, AD, DHD, DQD, R, D, TH, DVR, DVD, DD, DT = x
    # Equations on main variables
    dSdt = -alpha * gamma_t * S * I / N
    dEdt = alpha * gamma_t * S * I / N - r_i * E
    dIdt = r_i * E - r_d * I
    dARdt = r_d * (1 - p_dth_mod) * (1 - p_d) * I - r_ri * AR
    dDHRdt = r_d * (1 - p_dth_mod) * p_d * p_h * I - r_rh * DHR
    dDQRdt = r_d * (1 - p_dth_mod) * p_d * (1 - p_h) * I - r_ri * DQR
    dADdt = r_d * p_dth_mod * (1 - p_d) * I - r_dth * AD
    dDHDdt = r_d * p_dth_mod * p_d * p_h * I - r_dth * DHD
    dDQDdt = r_d * p_dth_mod * p_d * (1 - p_h) * I - r_dth * DQD
    dRdt = r_ri * (AR + DQR) + r_rh * DHR
    dDdt = r_dth * (AD + DQD + DHD)
    # Helper states (usually important for some kind of output)
    dTHdt = r_d * p_d * p_h * I
    dDVRdt = r_d * (1 - p_dth_mod) * p_d * p_h * p_v * I - r_rv * DVR
    dDVDdt = r_d * p_dth_mod * p_d * p_h * p_v * I - r_dth * DVD
    dDDdt = r_dth * (DHD + DQD)
    dDTdt = r_d * p_d * I
    return [
        dSdt, dEdt, dIdt, dARdt, dDHRdt, dDQRdt, dADdt, dDHDdt, dDQDdt,
        dRdt, dDdt, dTHdt, dDVRdt, dDVDdt, dDDdt, dDTdt
    ]
