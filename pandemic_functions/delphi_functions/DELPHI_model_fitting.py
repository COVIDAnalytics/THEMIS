# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu), Saksham Soni (sakshams@mit.edu)
import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import compress
from typing import Union, Type
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing, minimize
from scipy import stats
from datetime import datetime, timedelta
import time 
# from dateparser import parse
from dateutil.relativedelta import relativedelta

from pandemic_functions.pandemic_params import *
from pandemic_functions.delphi_functions.DELPHI_utils import *
from pandemic_functions.delphi_functions.DELPHI_model import model_covid

import os

past_parameters_default = pd.read_csv("pandemic_functions/pandemic_data/Parameters_Global_V2_annealing_20200420.csv")

def get_bounds_params_from_pastparams(
        optimizer: str, parameter_list: list, dict_default_reinit_parameters: dict, percentage_drift_lower_bound: float,
        default_lower_bound: float, dict_default_reinit_lower_bounds: dict, percentage_drift_upper_bound: float,
        default_upper_bound: float, dict_default_reinit_upper_bounds: dict,
        percentage_drift_lower_bound_annealing: float, default_lower_bound_annealing: float,
        percentage_drift_upper_bound_annealing: float, default_upper_bound_annealing: float,
        default_lower_bound_t_jump: float, default_upper_bound_t_jump: float, default_parameter_t_jump:float,
        default_lower_bound_std_normal: float, default_upper_bound_std_normal: float, default_parameter_std_normal: float
    ) -> list:
    """
    Generates the lower and upper bounds of the past parameters used as warm starts for the optimization process
    to predict with DELPHI: the output depends on the optimizer used (annealing or other, i.e. tnc or trust-constr)
    :param optimizer: optimizer used to obtain the DELPHI predictions
    :param parameter_list: list of all past parameter values for which we want to create bounds
    :param dict_default_reinit_parameters: dictionary with default values in case of reinitialization of parameters
    :param percentage_drift_lower_bound: percentage of drift allowed for the lower bound
    :param default_lower_bound: default lower bound value
    :param dict_default_reinit_lower_bounds: dictionary with lower bounds in case of reinitialization of parameters
    :param percentage_drift_upper_bound: percentage of drift allowed for the upper bound
    :param default_upper_bound: default upper bound value
    :param dict_default_reinit_upper_bounds: dictionary with upper bounds in case of reinitialization of parameters
    :param percentage_drift_lower_bound_annealing: percentage of drift allowed for the lower bound under annealing
    :param default_lower_bound_annealing: default lower bound value under annealing
    :param percentage_drift_upper_bound_annealing: percentage of drift allowed for the upper bound under annealing
    :param default_upper_bound_annealing: default upper bound value under annealing
    :param default_lower_bound_jump: default lower bound value for the jump parameter
    :param default_upper_bound_jump: default upper bound value for the jump parameter
    :param default_lower_bound_std_normal: default lower bound value for the normal standard deviation parameter
    :param default_upper_bound_std_normal: default upper bound value for the normal standard deviation parameter
    :return: a list of bounds for all the optimized parameters based on the optimizer and pre-fixed parameters
    """
    if optimizer in ["tnc", "trust-constr"]:
        # Allowing a drift for parameters
        alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal = parameter_list
        parameter_list = [
            max(alpha, dict_default_reinit_parameters["alpha"]),
            days,
            max(r_s, dict_default_reinit_parameters["r_s"]),
            max(min(r_dth, 1), dict_default_reinit_parameters["r_dth"]),
            max(min(p_dth, 1), dict_default_reinit_parameters["p_dth"]),
            max(r_dthdecay, dict_default_reinit_parameters["r_dthdecay"]),
            max(k1, dict_default_reinit_parameters["k1"]),
            max(k2, dict_default_reinit_parameters["k2"]),
            max(jump, dict_default_reinit_parameters["jump"]),
            max(t_jump, dict_default_reinit_parameters["t_jump"]),
            max(std_normal, dict_default_reinit_parameters["std_normal"]),
        ]
        param_list_lower = [x - max(percentage_drift_lower_bound * abs(x), default_lower_bound) for x in parameter_list]
        (
            alpha_lower, days_lower, r_s_lower, r_dth_lower, p_dth_lower, r_dthdecay_lower,
            k1_lower, k2_lower, jump_lower, t_jump_lower, std_normal_lower
        ) = param_list_lower
        param_list_lower = [
            max(alpha_lower, dict_default_reinit_lower_bounds["alpha"]),
            days_lower,
            max(r_s_lower, dict_default_reinit_lower_bounds["r_s"]),
            max(min(r_dth_lower, 1), dict_default_reinit_lower_bounds["r_dth"]),
            max(min(p_dth_lower, 1), dict_default_reinit_lower_bounds["p_dth"]),
            max(r_dthdecay_lower, dict_default_reinit_lower_bounds["r_dthdecay"]),
            max(k1_lower, dict_default_reinit_lower_bounds["k1"]),
            max(k2_lower, dict_default_reinit_lower_bounds["k2"]),
            max(jump_lower, dict_default_reinit_lower_bounds["jump"]),
            max(t_jump_lower, dict_default_reinit_lower_bounds["t_jump"]),
            max(std_normal_lower, dict_default_reinit_lower_bounds["std_normal"]),
        ]
        param_list_upper = [
            x + max(percentage_drift_upper_bound * abs(x), default_upper_bound) for x in parameter_list
        ]
        (
            alpha_upper, days_upper, r_s_upper, r_dth_upper, p_dth_upper, r_dthdecay_upper,
            k1_upper, k2_upper, jump_upper, t_jump_upper, std_normal_upper
        ) = param_list_upper
        param_list_upper = [
            max(alpha_upper, dict_default_reinit_upper_bounds["alpha"]),
            days_upper,
            max(r_s_upper, dict_default_reinit_upper_bounds["r_s"]),
            max(min(r_dth_upper, 1), dict_default_reinit_upper_bounds["r_dth"]),
            max(min(p_dth_upper, 1), dict_default_reinit_upper_bounds["p_dth"]),
            max(r_dthdecay_upper, dict_default_reinit_upper_bounds["r_dthdecay"]),
            max(k1_upper, dict_default_reinit_upper_bounds["k1"]),
            max(k2_upper, dict_default_reinit_upper_bounds["k2"]),
            max(jump_upper, dict_default_reinit_upper_bounds["jump"]),
            max(t_jump_upper, dict_default_reinit_upper_bounds["t_jump"]),
            max(std_normal_upper, dict_default_reinit_upper_bounds["std_normal"]),
        ]
    elif optimizer == "annealing":  # Annealing procedure for global optimization
        alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal = parameter_list
        parameter_list = [
            max(alpha, dict_default_reinit_parameters["alpha"]),
            days,
            max(r_s, dict_default_reinit_parameters["r_s"]),
            max(min(r_dth, 1), dict_default_reinit_parameters["r_dth"]),
            max(min(p_dth, 1), dict_default_reinit_parameters["p_dth"]),
            max(r_dthdecay, dict_default_reinit_parameters["r_dthdecay"]),
            max(k1, dict_default_reinit_parameters["k1"]),
            max(k2, dict_default_reinit_parameters["k2"]),
            max(jump, dict_default_reinit_parameters["jump"]),
            max(t_jump, dict_default_reinit_parameters["t_jump"]),
            max(std_normal, dict_default_reinit_parameters["std_normal"]),
        ]
        param_list_lower = [
            x - max(percentage_drift_lower_bound_annealing * abs(x), default_lower_bound_annealing) for x in
            parameter_list
        ]
        (
            alpha_lower, days_lower, r_s_lower, r_dth_lower, p_dth_lower, r_dthdecay_lower,
            k1_lower, k2_lower, jump_lower, t_jump_lower, std_normal_lower
        ) = param_list_lower
        param_list_lower = [
            max(alpha_lower, dict_default_reinit_lower_bounds["alpha"]),
            days_lower,
            max(r_s_lower, dict_default_reinit_lower_bounds["r_s"]),
            max(min(r_dth_lower, 1), dict_default_reinit_lower_bounds["r_dth"]),
            max(min(p_dth_lower, 1), dict_default_reinit_lower_bounds["p_dth"]),
            max(r_dthdecay_lower, dict_default_reinit_lower_bounds["r_dthdecay"]),
            max(k1_lower, dict_default_reinit_lower_bounds["k1"]),
            max(k2_lower, dict_default_reinit_lower_bounds["k2"]),
            max(jump_lower, dict_default_reinit_lower_bounds["jump"]),
            max(t_jump_lower, dict_default_reinit_lower_bounds["t_jump"]),
            max(std_normal_lower, dict_default_reinit_lower_bounds["std_normal"]),
        ]
        param_list_upper = [
            x + max(percentage_drift_upper_bound_annealing * abs(x), default_upper_bound_annealing) for x in
            parameter_list
        ]
        (
            alpha_upper, days_upper, r_s_upper, r_dth_upper, p_dth_upper, r_dthdecay_upper,
            k1_upper, k2_upper, jump_upper, t_jump_upper, std_normal_upper
        ) = param_list_upper
        param_list_upper = [
            max(alpha_upper, dict_default_reinit_upper_bounds["alpha"]),
            days_upper,
            max(r_s_upper, dict_default_reinit_upper_bounds["r_s"]),
            max(min(r_dth_upper, 1), dict_default_reinit_upper_bounds["r_dth"]),
            max(min(p_dth_upper, 1), dict_default_reinit_upper_bounds["p_dth"]),
            max(r_dthdecay_upper, dict_default_reinit_upper_bounds["r_dthdecay"]),
            max(k1_upper, dict_default_reinit_upper_bounds["k1"]),
            max(k2_upper, dict_default_reinit_upper_bounds["k2"]),
            max(jump_upper, dict_default_reinit_upper_bounds["jump"]),
            max(t_jump_upper, dict_default_reinit_upper_bounds["t_jump"]),
            max(std_normal_upper, dict_default_reinit_upper_bounds["std_normal"]),
        ]
        param_list_lower[9] = default_lower_bound_t_jump  # jump lower bound
        param_list_upper[9] = default_upper_bound_t_jump  # jump upper bound
        parameter_list[9] = default_parameter_t_jump # jump parameter
        parameter_list[10] = default_parameter_std_normal # std_normal parameter
        param_list_lower[10] = default_lower_bound_std_normal  # std_normal lower bound
        param_list_upper[10] = default_upper_bound_std_normal  # std_normal upper bound
    else:
        raise ValueError(f"Optimizer {optimizer} not supported in this implementation so can't generate bounds")

    bounds_params = [(lower, upper) for lower, upper in zip(param_list_lower, param_list_upper)]
    return parameter_list, bounds_params


## TODO
## - ability to fit on data from T_start instead of date_day_since100
def solve_and_predict_area(
        region: str,
        yesterday: str,
        past_parameters,
        totalcases: Union[pd.DataFrame, Type[None]]=None,
        start_date: Union[str, Type[None]]=None,
        end_date: Union[str, Type[None]]=None,
        optimization_method: str='annealing'
    ):
    """
    Fit parameters for a region based on the parameters till the date `yesterday`

    :param tuple_region: tuple corresponding to (continent:str, country:str, province:str)
    :param yesterday: string corresponding to the date from which the model will read the previous parameters. The
    format has to be 'YYYYMMDD'
    :param past_parameters: Parameters from `yesterday` used as a starting point for the fitting process
    :start_date: string for the date from when the pandemic will be modelled (format should be 'YYYY-MM-DD')
    :end_date: string for the date till when the predictions will be made (format should be 'YYYY-MM-DD')
    :return: either None if can't optimize (either less than 100 cases or less than 7 days with 100 cases) or a tuple
    with 3 dataframes related to that `tuple_region` (parameters df, predictions since yesterday+1, predictions since
    first day with 100 cases) and a scipy.optimize object (OptimizeResult) that contains the predictions for all
    16 states of the model (and some other information that isn't used)
    """
    time_entering = time.time()
    country, province = region_symbol_country_dict[region]
    continent = region_symbol_continent_dict[region]
    country_sub = country.replace(" ", "_")
    province_sub = province.replace(" ", "_")

    if totalcases is None:
        assert os.path.exists(f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"), \
            f"Cases data not found for region {region}"
        totalcases = pd.read_csv(
                f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"
            )
    
    # set the initial parameters
    if past_parameters is not None and len(past_parameters[(past_parameters.Country == country) & (past_parameters.Province == province)]) > 0:
        # if past parameters are available
        parameter_list_total = past_parameters[  # find saved parameters for the region
            (past_parameters.Country == country) &
            (past_parameters.Province == province)
            ].reset_index(drop=True)

        parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
        parameter_list = parameter_list_line[5:] # learned parameters
        date_day_since100 = pd.to_datetime(parameter_list_line[3])
        validcases = totalcases[
            (totalcases.day_since100 >= 0) &
            (totalcases.date <= str((pd.to_datetime(yesterday) + timedelta(days=1)).date()))
            ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
        bounds_params = get_bounds_params_from_pastparams(
            optimizer=optimization_method,
            parameter_list=parameter_list,
            dict_default_reinit_parameters=dict_default_reinit_parameters,
            percentage_drift_lower_bound=percentage_drift_lower_bound,
            default_lower_bound=default_lower_bound,
            dict_default_reinit_lower_bounds=dict_default_reinit_lower_bounds,
            percentage_drift_upper_bound=percentage_drift_upper_bound,
            default_upper_bound=default_upper_bound,
            dict_default_reinit_upper_bounds=dict_default_reinit_upper_bounds,
            percentage_drift_lower_bound_annealing=percentage_drift_lower_bound_annealing,
            default_lower_bound_annealing=default_lower_bound_annealing,
            percentage_drift_upper_bound_annealing=percentage_drift_upper_bound_annealing,
            default_upper_bound_annealing=default_upper_bound_annealing,
            default_lower_bound_jump=default_lower_bound_jump,
            default_upper_bound_jump=default_upper_bound_jump,
            default_lower_bound_std_normal=default_lower_bound_std_normal,
            default_upper_bound_std_normal=default_upper_bound_std_normal,
        )
    else:
        # Otherwise use established lower/upper bounds
        date_day_since100 = pd.to_datetime(totalcases.loc[totalcases.day_since100 == 0, "date"].iloc[-1])
        validcases = totalcases[
            (totalcases.day_since100 >= 0) &
            (totalcases.date <= str((pd.to_datetime(yesterday) + timedelta(days=1)).date()))
            ][["day_since100", "case_cnt", "death_cnt"]].reset_index(drop=True)
        parameter_list, bounds_params = default_parameter_list, default_bounds_params
    
    # Now we start the modeling part:
    assert len(validcases) > validcases_threshold, \
        f"Not enough historical data (less than a week) for Country={country} and Province={province}"
    
    PopulationT = global_populations[
        (global_populations.Country == country) & (global_populations.Province == province)
        ].pop2016.iloc[-1]
    # We do not scale
    N = PopulationT
    PopulationI = validcases.loc[0, "case_cnt"]
    PopulationR = validcases.loc[0, "death_cnt"] * 5
    PopulationD = validcases.loc[0, "death_cnt"]
    
    # Currently fit on alpha, a and b, r_dth,
    # & initial condition of exposed state and infected state
    # Maximum timespan of prediction, defaulted to go to 12/31/2020
    maxT = (default_maxT - date_day_since100).days + 1 if end_date is None else (pd.to_datetime(end_date) - date_day_since100).days + 1

    ## Fit on Total Cases
    t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
    validcases_nondeath = validcases["case_cnt"].tolist()
    validcases_death = validcases["death_cnt"].tolist()
    # balance: Ratio of Fitting between cases and deaths
    balance = validcases_nondeath[-1] / max(validcases_death[-1], 10) / 3
    cases_data_fit = validcases_nondeath
    deaths_data_fit = validcases_death
    GLOBAL_PARAMS_FIXED = (
        N, PopulationR, PopulationD, PopulationI, p_d, p_h, p_v
    )

    def model_covid_predictions(t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, 
            k1, k2, jump, t_jump, std_normal):
        return model_covid(t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, 
            k1, k2, jump, t_jump, std_normal, N, None, None, maxT)

    def residuals_totalcases(params):
        """
        returns the residuals for the prediction to fit the parameters
        params: (alpha, days, r_s, r_dth, p_dth, k1, k2), fitted parameters of the model
        """
        # Variables Initialization for the ODE system
        alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal = params
        params = (
            max(alpha, dict_default_reinit_parameters["alpha"]),
            days,
            max(r_s, dict_default_reinit_parameters["r_s"]),
            max(min(r_dth, 1), dict_default_reinit_parameters["r_dth"]),
            max(min(p_dth, 1), dict_default_reinit_parameters["p_dth"]),
            max(r_dthdecay, dict_default_reinit_parameters["r_dthdecay"]),
            max(k1, dict_default_reinit_parameters["k1"]),
            max(k2, dict_default_reinit_parameters["k2"]),
            max(jump, dict_default_reinit_parameters["jump"]),
            max(t_jump, dict_default_reinit_parameters["t_jump"]),
            max(std_normal, dict_default_reinit_parameters["std_normal"]),
        )
        x_0_cases = get_initial_conditions(
            params_fitted=params,
            global_params_fixed=GLOBAL_PARAMS_FIXED
        )
        x_sol_total = solve_ivp(
            fun=model_covid_predictions,
            y0=x_0_cases,
            t_span=[t_cases[0], t_cases[-1]],
            t_eval=t_cases,
            args=tuple(params)
        )
        x_sol = x_sol_total.y
        weights = list(range(1, len(cases_data_fit) + 1))
        weights = [( x / len(cases_data_fit) )**2 for x in weights]
        # weights[-15:] =[x + 50 for x in weights[-15:]]
        if x_sol_total.status == 0:
            residuals_value = sum(
                np.multiply((x_sol[15, :] - cases_data_fit) ** 2, weights)
                + balance * balance * np.multiply((x_sol[14, :] - deaths_data_fit) ** 2, weights)) + sum(
                np.multiply((x_sol[15, 7:] - x_sol[15, :-7] - cases_data_fit[7:] + cases_data_fit[:-7]) ** 2, weights[7:])
                + balance * balance * np.multiply((x_sol[14, 7:] - x_sol[14, :-7] - deaths_data_fit[7:] + deaths_data_fit[:-7]) ** 2, weights[7:])
                )
            return residuals_value
        else:
            residuals_value = 1e12
            return residuals_value

    if optimization_method == 'annealing':
        output = dual_annealing(residuals_totalcases, x0 = parameter_list, bounds = bounds_params)
    elif optimization_method == 'tnc':
        output = minimize(
            residuals_totalcases,
            parameter_list,
            method=dual_annealing,  # Can't use Nelder-Mead if I want to put bounds on the params
            bounds=bounds_params,
            options={'maxiter': max_iter}
        )
    best_params = output.x
    t_predictions = [i for i in range(maxT)]

    def solve_best_params_and_predict(optimal_params):
        # Variables Initialization for the ODE system
        alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal = optimal_params
        optimal_params = [
            max(alpha, dict_default_reinit_parameters["alpha"]),
            days,
            max(r_s, dict_default_reinit_parameters["r_s"]),
            max(min(r_dth, 1), dict_default_reinit_parameters["r_dth"]),
            max(min(p_dth, 1), dict_default_reinit_parameters["p_dth"]),
            max(r_dthdecay, dict_default_reinit_parameters["r_dthdecay"]),
            max(k1, dict_default_reinit_parameters["k1"]),
            max(k2, dict_default_reinit_parameters["k2"]),
            max(jump, dict_default_reinit_parameters["jump"]),
            max(t_jump, dict_default_reinit_parameters["t_jump"]),
            max(std_normal, dict_default_reinit_parameters["std_normal"]),
        ]
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

    mape_data = get_mape_data_fitting(cases_data_fit, deaths_data_fit, x_sol_final)
    df_parameters = create_parameters_dataframe(continent, country, province, date_day_since100, 
                                                mape_data, best_params)
    df_predictions_since_today, df_predictions_since_100 = create_datasets_predictions(
                            continent, country, province, date_day_since100, yesterday, x_sol_final)

    print(
        f"Finished predicting for Country={country} and Province={province} in " +
        f"{round(time.time() - time_entering, 2)} seconds"
    )
    
    return (
        df_parameters, df_predictions_since_today,
        df_predictions_since_100, output
    )

