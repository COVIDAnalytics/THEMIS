#%%

# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu), Saksham Soni (sakshams@mit.edu)
import sys
sys.path.append("Users/saksham/Research/COVIDAnalytics/DELPHI/")
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
from dateparser import parse
from dateutil.relativedelta import relativedelta
# from DELPHI_utils_V4_static import compute_mape, create_fitting_data_from_validcases, get_mape_data_fitting, DELPHIAggregations
from DELPHI_utils_V4_dynamic import (
    read_oxford_international_policy_data, get_normalized_policy_shifts_and_current_policy_all_countries,
    get_normalized_policy_shifts_and_current_policy_us_only, read_policy_data_us_only
)
from DELPHI_params_V4 import (
    fitting_start_date,
    date_MATHEMATICA, validcases_threshold_policy, default_dict_normalized_policy_gamma,
    IncubeD, RecoverID, RecoverHD, DetectD, VentilatedD,
    default_maxT_policies, p_v, future_policies, future_times
)
p_d=0.2
p_h=0.03
import yaml
import os
import argparse

#%%

with open("config.yml", "r") as ymlfile:
    CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)
CONFIG_FILEPATHS = CONFIG["filepaths"]
yesterday = "".join(str(datetime.now().date() - timedelta(days=1)).split("-"))
parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--run_config', '-rc', type=str, required=True,
#     help="specify relative path for the run config YAML file"
# )
# arguments = parser.parse_args()
# with open(arguments.run_config, "r") as ymlfile:
#     RUN_CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)

#%%
with open('run_configs/run-config.yml', "r") as ymlfile:
    RUN_CONFIG = yaml.load(ymlfile, Loader=yaml.BaseLoader)

#%%

USER_RUNNING = RUN_CONFIG["arguments"]["user"]
OPTIMIZER = RUN_CONFIG["arguments"]["optimizer"]
GET_CONFIDENCE_INTERVALS = bool(int(RUN_CONFIG["arguments"]["confidence_intervals"]))
SAVE_TO_WEBSITE = bool(int(RUN_CONFIG["arguments"]["website"]))
SAVE_SINCE100_CASES = bool(int(RUN_CONFIG["arguments"]["since100case"]))
PATH_TO_FOLDER_DANGER_MAP = CONFIG_FILEPATHS["danger_map"][USER_RUNNING]
PATH_TO_DATA_SANDBOX = CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING]
PATH_TO_WEBSITE_PREDICTED = CONFIG_FILEPATHS["website"][USER_RUNNING]
policy_data_countries = read_oxford_international_policy_data(yesterday=yesterday)
policy_data_us_only = read_policy_data_us_only(filepath_data_sandbox=CONFIG_FILEPATHS["data_sandbox"][USER_RUNNING])
popcountries = pd.read_csv(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Population_Global.csv")
df_initial_states = pd.read_csv(
    PATH_TO_DATA_SANDBOX + f"predicted/raw_predictions/Predicted_model_state_V4_{fitting_start_date}.csv"
)
subname_parameters_file = None
if OPTIMIZER == "tnc":
    subname_parameters_file = "Global_V4"
elif OPTIMIZER == "annealing":
    subname_parameters_file = "Global_V4_annealing"
elif OPTIMIZER == "trust-constr":
    subname_parameters_file = "Global_V4_trust"
else:
    raise ValueError("Optimizer not supported in this implementation")
yesterday = "20210305"
past_parameters = pd.read_csv(
    PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_{subname_parameters_file}_{yesterday}.csv"
)
if pd.to_datetime(yesterday) < pd.to_datetime(date_MATHEMATICA):
    param_MATHEMATICA = True
else:
    param_MATHEMATICA = False
# True if we use the Mathematica run parameters, False if we use those from Python runs
# This is because the past_parameters dataframe's columns are not in the same order in both cases



def get_initial_conditions(params_fitted: tuple, global_params_fixed: tuple) -> list:
    """
    Generates the initial conditions for the DELPHI model based on global fixed parameters (mostly populations and some
    constant rates) and fitted parameters (the internal parameters k1 and k2)
    :param params_fitted: tuple of parameters being fitted, mostly interested in k1 and k2 here (parameters 7 and 8)
    :param global_params_fixed: tuple of fixed and constant parameters for the model defined a while ago
    :return: a list of initial conditions for all 16 states of the DELPHI model
    """
    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3, p_d, p_h = params_fitted 
    N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_v = global_params_fixed

    PopulationR = min(R_upperbound - 1, min(int(R_0*p_d), R_heuristic))
    PopulationCI = (PopulationI - PopulationD - PopulationR)*k3

    S_0 = (
        (N - PopulationCI / p_d)
        - (PopulationCI / p_d * (k1 + k2))
        - (PopulationR / p_d)
        - (PopulationD / p_d)
    )
    E_0 = PopulationCI / p_d * k1
    I_0 = PopulationCI / p_d * k2
    UR_0 = (PopulationCI / p_d - PopulationCI) * (1 - p_dth)
    DHR_0 = (PopulationCI * p_h) * (1 - p_dth)
    DQR_0 = PopulationCI * (1 - p_h) * (1 - p_dth)
    UD_0 = (PopulationCI / p_d - PopulationCI) * p_dth
    DHD_0 = PopulationCI * p_h * p_dth
    DQD_0 = PopulationCI * (1 - p_h) * p_dth
    R_0 = PopulationR / p_d
    D_0 = PopulationD / p_d
    TH_0 = PopulationCI * p_h
    DVR_0 = (PopulationCI * p_h * p_v) * (1 - p_dth)
    DVD_0 = (PopulationCI * p_h * p_v) * p_dth
    DD_0 = PopulationD
    DT_0 = PopulationI
    x_0_cases = [
        S_0, E_0, I_0, UR_0, DHR_0, DQR_0, UD_0, DHD_0, DQD_0, R_0,
        D_0, TH_0, DVR_0, DVD_0, DD_0, DT_0,
    ]
    return x_0_cases


# %%

def run_delphi_policy_scenario(policy, country):
    # TODO implement us policies
    country_sub = country.replace(' ', '_')
    province=province_sub="None"
    past_parameters = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"predicted/Parameters_{subname_parameters_file}_{yesterday}.csv"
    )
    policy_data_countries = read_oxford_international_policy_data(yesterday=yesterday)
    
    # Get the policies shifts from the CART tree to compute different values of gamma(t)
    # Depending on the policy in place in the future to affect predictions
    dict_normalized_policy_gamma_countries, dict_current_policy_countries = (
        get_normalized_policy_shifts_and_current_policy_all_countries(
            policy_data_countries=policy_data_countries,
            past_parameters=past_parameters,
        )
    )
    # Setting same value for these 2 policies because of the inherent structure of the tree
    dict_normalized_policy_gamma_countries[future_policies[3]] = dict_normalized_policy_gamma_countries[future_policies[5]]
    dict_current_policy_international = dict_current_policy_countries.copy()
    dict_normalized_policy_gamma_countries = default_dict_normalized_policy_gamma

    if os.path.exists(PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_None.csv"):
        totalcases = pd.read_csv(
            PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_None.csv"
        )
    else:
        raise FileNotFoundError(f"Can not find file - processed/Global/Cases_{country_sub}_None.csv for actual polcy outcome")

    parameter_list_total = past_parameters[past_parameters.Country == country]
    totalcases = pd.read_csv(
        PATH_TO_FOLDER_DANGER_MAP + f"processed/Global/Cases_{country_sub}_{province_sub}.csv"
    )

    if len(parameter_list_total) > 0:
        parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
        parameter_list = parameter_list_line[5:]
        date_day_since100 = pd.to_datetime(parameter_list_line[3])
        # Allowing a 5% drift for states with past predictions, starting in the 5th position are the parameters
        start_date = date_day_since100
        validcases = totalcases[
            (totalcases.date >= str(policy.start_date))
            & (totalcases.date <= str(policy.end_date))
        ][["day_since100", "case_cnt", "death_cnt", "total_hospitalization", "people_vaccinated", "people_fully_vaccinated"]].reset_index(drop=True)
    else:
        print(f"Must have past parameters for {country} and {province}")
        return 0, 0, 0

    # Now we start the modeling part:
    if len(validcases) > validcases_threshold_policy:
        PopulationT = popcountries[
            (popcountries.Country == country) & (popcountries.Province == province)
        ].pop2016.iloc[-1]
        N = PopulationT
        PopulationI = validcases.loc[0, "case_cnt"]
        PopulationD = validcases.loc[0, "death_cnt"]
        R_0 = validcases.loc[0, "death_cnt"] * 5 if validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]> validcases.loc[0, "death_cnt"] * 5 else 0
        cases_t_14days = totalcases[totalcases.date >= str(start_date- pd.Timedelta(14, 'D'))]['case_cnt'].values[0]
        deaths_t_9days = totalcases[totalcases.date >= str(start_date - pd.Timedelta(9, 'D'))]['death_cnt'].values[0]
        R_upperbound = validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]
        R_heuristic = cases_t_14days - deaths_t_9days

        """
        Fixed Parameters based on meta-analysis:
        p_h: Hospitalization Percentage
        RecoverHD: Average Days until Recovery
        VentilationD: Number of Days on Ventilation for Ventilated Patients
        maxT: Maximum # of Days Modeled
        p_d: Percentage of True Cases Detected
        p_v: Percentage of Hospitalized Patients Ventilated,
        balance: Regularization coefficient between cases and deaths
        """
        maxT = (default_maxT_policies - date_day_since100).days + 1
        t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
        # balance, cases_data_fit, deaths_data_fit, hosp_balance, hosp_data_fit = create_fitting_data_from_validcases(validcases)
        GLOBAL_PARAMS_FIXED = (N, R_upperbound, R_heuristic, R_0, PopulationD, PopulationI, p_v, p_d, p_h)
        best_params = parameter_list
        t_predictions = [i for i in range(maxT)]
        for future_policy in future_policies:
            for future_time in future_times:
                def model_covid_predictions(
                        t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal, k3
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
                    gamma_t_future = (
                            (2 / np.pi) * np.arctan(-(t_cases[-1] + future_time - days) / 20 * r_s) + 1 +
                            jump * np.exp(-(t_cases[-1] + future_time - t_jump)**2 / (2 * std_normal ** 2))
                    )
                    p_dth_mod = (2 / np.pi) * (p_dth - 0.01) * (np.arctan(- t / 20 * r_dthdecay) + np.pi / 2) + 0.01
                    if t > t_cases[-1] + future_time:
                        normalized_gamma_future_policy = dict_normalized_policy_gamma_countries[future_policy]
                        normalized_gamma_current_policy = dict_normalized_policy_gamma_countries[
                            dict_current_policy_international[(country, province)]
                        ]
                        epsilon = 1e-4
                        gamma_t = gamma_t + min(
                            (2 - gamma_t_future) / (1 - normalized_gamma_future_policy + epsilon),
                            (gamma_t_future / normalized_gamma_current_policy) *
                            (normalized_gamma_future_policy - normalized_gamma_current_policy)
                        )

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


                def solve_best_params_and_predict(optimal_params):
                    # Variables Initialization for the ODE system
                    x_0_cases = get_initial_conditions(
                        params_fitted=optimal_params,
                        global_params_fixed=GLOBAL_PARAMS_FIXED
                    )
                    x_sol_best = solve_ivp(
                        fun=model_covid_predictions,
                        y0=x_0_cases,
                        t_span=[t_predictions[0], t_predictions[-1]],
                        t_eval=t_predictions,
                        args=tuple(optimal_params),
                    ).y
                    return x_sol_best


                x_sol_final = solve_best_params_and_predict(best_params)

    return 0, 0, 0
