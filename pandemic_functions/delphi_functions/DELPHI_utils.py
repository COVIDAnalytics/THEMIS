# common utils for DELPHI functions
# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu), Saksham Soni (sakshams@mit.edu)

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_inv(x):
    if x==1 or x==0:
        raise ZeroDivisionError
    # EPS = 1e-6
    return np.log(x/(1-x))

sigmoid_inv_np = np.vectorize(sigmoid_inv)

def gamma_t(day: datetime, params_list: list):
    """
    Computes values of our gamma(t) function with the exponential jump
    :param day: day on which we want to compute the value of gamma(t)
    :param params_list: list with format [dsd, median_day_of_action, rate_of_action, jump, t_jump, std_normal]
    :return: value of gamma(t) on that day and with the input parameters
    """
    dsd, median_day_of_action, rate_of_action, jump, t_jump, std_normal = params_list
    t = (day - pd.to_datetime(dsd)).days
    gamma_t = (
        (2 / np.pi) * np.arctan(-(t - median_day_of_action) / 20 * rate_of_action) + 1 +
        jump * np.exp(-(t - t_jump)**2 /(2 * std_normal ** 2))
    )
    return gamma_t

def get_initial_conditions(params_fitted: tuple, global_params_fixed: tuple) -> list:
    """
    Generates the initial conditions for the DELPHI model based on global fixed parameters (mostly populations and some
    constant rates) and fitted parameters (the internal parameters k1 and k2)
    :param params_fitted: tuple of parameters being fitted, mostly interested in k1 and k2 here (parameters 7 and 8)
    :param global_params_fixed: tuple of fixed and constant parameters for the model defined a while ago
    :return: a list of initial conditions for all 16 states of the DELPHI model
    """
    alpha, days, r_s, r_dth, p_dth, r_dthdecay, k1, k2, jump, t_jump, std_normal  = params_fitted 
    N, PopulationR, PopulationD, PopulationI, p_v, p_d, p_h = global_params_fixed

    PopulationCI = (PopulationI - PopulationD - PopulationR)

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


def compute_mape(y_true: list, y_pred: list) -> float:
    """
    Compute the Mean Absolute Percentage Error (MAPE) between two lists of values
    :param y_true: list of true historical values
    :param y_pred: list of predicted values
    :return: a float corresponding to the MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred)[y_true > 0] / y_true[y_true > 0])) * 100
    return mape

def compute_mse(y_true: list, y_pred: list) -> float:
    """
    Compute the Mean Squared Error between two lists
    :param y_true: list of true historical values
    :param y_pred: list of predicted values
    :return: a float, corresponding to the MSE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse)


def compute_mae(y_true: list, y_pred: list) -> float:
    """
    Compute the Mean Absolute Error (MAE) between two lists of values
    :param y_true: list of true historical values
    :param y_pred: list of predicted values
    :return: a tuple of floats, corresponding to (MAE, MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs((y_true - y_pred)))
    return mae

def get_mape_data_fitting(cases_data_fit: list, deaths_data_fit: list, x_sol_final: np.array) -> float:
    """
    Computes MAPE on cases & deaths (averaged) either on last 15 days of historical data (if there are more than 15)
    or exactly the number of days in the historical data (if less than 15)
    :param cases_data_fit: list, contains data used to fit on number of cases
    :param deaths_data_fit: list, contains data used to fit on number of deaths
    :param x_sol_final: numpy array, contains the predicted solution by the DELPHI model for all 16 states
    :return: a float corresponding to the average MAPE on cases and deaths on a given period of time (15 days is default
    but otherwise the number of days available in the historical data)
    """
    if len(cases_data_fit) > 15:  # In which case we can compute MAPE on last 15 days
        mape_data = (
                compute_mape(
                    cases_data_fit[-15:],
                    x_sol_final[15, len(cases_data_fit) - 15: len(cases_data_fit)],
                ) + compute_mape(
                    deaths_data_fit[-15:],
                    x_sol_final[14, len(deaths_data_fit) - 15: len(deaths_data_fit)],
                )
        ) / 2
    else:  # We take MAPE on all available previous days (less than 15)
        mape_data = (
                compute_mape(cases_data_fit, x_sol_final[15, : len(cases_data_fit)])
                + compute_mape(deaths_data_fit, x_sol_final[14, : len(deaths_data_fit)])
        ) / 2

    return mape_data

def get_mape_test_data(cases_data_test: list, deaths_data_test: list, cases_prediction: list, deaths_prediction: list):
    """
    Calculate mape over test data
    :param cases_data_test: list, contains data of true number of cases
    :param deaths_data_test: list, contains data of true number of deaths
    :param cases_prediction: list, contains prediction for cases
    :param deaths_prediction: list, contains prediction for deaths
    :return: a tuple, (mape_cases: float, mape_deaths: float)
    """
    mape_cases = compute_mape(cases_data_test, cases_prediction) 
    mape_deaths = compute_mape(deaths_data_test, deaths_prediction)
    return mape_cases, mape_deaths

def create_parameters_dataframe(continent:str, country:str, province:str,
        date_day_since100: datetime, mape: float, best_params: np.array) -> pd.DataFrame:
    """
    Creates the parameters dataset with the results from the optimization and the pre-computed MAPE
    """

    df_parameters = pd.DataFrame(
        {
            "Continent": [continent],
            "Country": [country],
            "Province": [province],
            "Data Start Date": [date_day_since100],
            "MAPE": [mape],
            "Infection Rate": [best_params[0]],
            "Median Day of Action": [best_params[1]],
            "Rate of Action": [best_params[2]],
            "Rate of Death": [best_params[3]],
            "Mortality Rate": [best_params[4]],
            "Rate of Mortality Rate Decay": [best_params[5]],
            "Internal Parameter 1": [best_params[6]],
            "Internal Parameter 2": [best_params[7]],
            "Jump Magnitude": [best_params[8]],
            "Jump Time": [best_params[9]],
            "Jump Decay": [best_params[10]],
        }
    )
    return df_parameters

def get_predictions_from_solution(x_sol_final: np.array): 
    """
    Returns the list of predictions from the final result of solve_ivp function
    """
    # Predictions
    total_detected = x_sol_final[15, :]  # DT
    total_detected = [int(round(x, 0)) for x in total_detected]
    active_cases = (
            x_sol_final[4, :]
            + x_sol_final[5, :]
            + x_sol_final[7, :]
            + x_sol_final[8, :]
    )  # DHR + DQR + DHD + DQD
    active_cases = [int(round(x, 0)) for x in active_cases]
    active_hospitalized = (
            x_sol_final[4, :] + x_sol_final[7, :]
    )  # DHR + DHD
    active_hospitalized = [int(round(x, 0)) for x in active_hospitalized]
    cumulative_hospitalized = x_sol_final[11, :]  # TH
    cumulative_hospitalized = [int(round(x, 0)) for x in cumulative_hospitalized]
    total_detected_deaths = x_sol_final[14, :]  # DD
    total_detected_deaths = [int(round(x, 0)) for x in total_detected_deaths]
    active_ventilated = (
            x_sol_final[12, :] + x_sol_final[13, :]
    )  # DVR + DVD
    active_ventilated = [int(round(x, 0)) for x in active_ventilated]

    return (total_detected, active_cases, active_hospitalized, cumulative_hospitalized, 
            total_detected_deaths, active_ventilated)

def create_datasets_predictions(continent:str, country:str, province:str,
        date_day_since100: datetime, yesterday: str,
        x_sol_final: np.array) -> (pd.DataFrame, pd.DataFrame):
    """
    Creates two dataframes with the predictions of the DELPHI model, the first one since the 
    day of the prediction, the second since the day the area had 100 cases
    :return: tuple of dataframes with predictions from DELPHI model
    """
    today_date = pd.to_datetime(yesterday) + timedelta(days=1)
    n_days_btw_today_since_100 = (today_date - date_day_since100).days
    n_days_since_today = x_sol_final.shape[1] - n_days_btw_today_since_100
    all_dates_since_today = [
        str((today_date + timedelta(days=i)).date())
        for i in range(n_days_since_today)
    ]
    total_detected, active_cases, active_hospitalized, cumulative_hospitalized, \
        total_detected_deaths, active_ventilated = get_predictions_from_solution(x_sol_final)

    # Generation of the dataframe since today
    df_predictions_since_today_cont_country_prov = pd.DataFrame(
        {
            "Continent": [continent for _ in range(n_days_since_today)],
            "Country": [country for _ in range(n_days_since_today)],
            "Province": [province for _ in range(n_days_since_today)],
            "Day": all_dates_since_today,
            "Total Detected": total_detected[n_days_btw_today_since_100:],
            "Active": active_cases[n_days_btw_today_since_100:],
            "Active Hospitalized": active_hospitalized[n_days_btw_today_since_100:],
            "Cumulative Hospitalized": cumulative_hospitalized[
                                        n_days_btw_today_since_100:
                                        ],
            "Total Detected Deaths": total_detected_deaths[
                                        n_days_btw_today_since_100:
                                        ],
            "Active Ventilated": active_ventilated[n_days_btw_today_since_100:],
        }
    )

    # Generation of the dataframe from the day since 100th case
    all_dates_since_100 = [
        str((date_day_since100 + timedelta(days=i)).date())
        for i in range(x_sol_final.shape[1])
    ]
    df_predictions_since_100_cont_country_prov = pd.DataFrame(
        {
            "Continent": [continent for _ in range(len(all_dates_since_100))],
            "Country": [country for _ in range(len(all_dates_since_100))],
            "Province": [province for _ in range(len(all_dates_since_100))],
            "Day": all_dates_since_100,
            "Total Detected": total_detected,
            "Active": active_cases,
            "Active Hospitalized": active_hospitalized,
            "Cumulative Hospitalized": cumulative_hospitalized,
            "Total Detected Deaths": total_detected_deaths,
            "Active Ventilated": active_ventilated,
        }
    )
    return (
        df_predictions_since_today_cont_country_prov,
        df_predictions_since_100_cont_country_prov,
    )

def make_increasing(sequence: list) -> list:
    """
    Used to force the Confidence Intervals generated for DELPHI to be always increasing
    :param sequence: list, sequence of values
    :return: list, forcefully increasing sequence of values
    """
    for i in range(len(sequence)):
        sequence[i] = max(sequence[i], sequence[max(i-1, 0)])
    return sequence

## TODO
## - test this function
def create_datasets_with_confidence_intervals(
        continent:str, country:str, province:str,
        date_day_since100: datetime, yesterday: str,
        x_sol_final: np.array,
        cases_data_fit: list,
        deaths_data_fit: list,
        past_prediction_file: str = "I://covid19orc//danger_map//predicted//Global_V2_20200720.csv",
        past_prediction_date: str = "2020-07-04",
        q: float = 0.5,
    ) -> (pd.DataFrame, pd.DataFrame):
    """
    Generates the prediction datasets from the date with 100 cases and from the day of running, including columns
    containing Confidence Intervals used in the website for cases and deaths
    :param continent, str
    :param country, str
    :param province, str
    :param date_day_since100: datetime, date to start the data from
    :param x_sol_final: np.array, final predictions from the model
    :param cases_data_fit: list, contains data used to fit on number of cases
    :param deaths_data_fit: list, contains data used to fit on number of deaths
    :param past_prediction_file: past prediction file's path for CI generation
    :param past_prediction_date: past prediction's date for CI generation
    :param q: quantile used for the CIs
    :return: tuple of dataframes (since day of optimization & since 100 cases in the area) with predictions and
    confidence intervals
    """
    today_date = pd.to_datetime(yesterday) + timedelta(days=1)
    n_days_btw_today_since_100 = (today_date - date_day_since100).days
    n_days_since_today = x_sol_final.shape[1] - n_days_btw_today_since_100
    all_dates_since_today = [
        str((today_date + timedelta(days=i)).date())
        for i in range(n_days_since_today)
    ]
    total_detected, active_cases, active_hospitalized, cumulative_hospitalized, \
        total_detected_deaths, active_ventilated = get_predictions_from_solution(x_sol_final)

    # using past predictions
    past_predictions = pd.read_csv(past_prediction_file)
    past_predictions = (
        past_predictions[
            (past_predictions["Day"] > past_prediction_date)
            & (past_predictions["Country"] == country)
            & (past_predictions["Province"] == province)
            ]
    ).sort_values("Day")
    if len(past_predictions) > 0:
        known_dates_since_100 = [
            str((date_day_since100 + timedelta(days=i)).date())
            for i in range(len(cases_data_fit))
        ]
        cases_data_fit_past = [
            y
            for x, y in zip(known_dates_since_100, cases_data_fit)
            if x > past_prediction_date
        ]
        deaths_data_fit_past = [
            y
            for x, y in zip(known_dates_since_100, deaths_data_fit)
            if x > past_prediction_date
        ]
        total_detected_past = past_predictions["Total Detected"].values[
                                : len(cases_data_fit_past)
                                ]
        total_detected_deaths_past = past_predictions[
                                            "Total Detected Deaths"
                                        ].values[: len(deaths_data_fit_past)]
        residual_cases_lb = np.sqrt(
            np.mean(
                [(x - y) ** 2 for x, y in zip(cases_data_fit_past, total_detected_past)]
            )
        ) * stats.norm.ppf(0.5 - q / 2)
        residual_cases_ub = np.sqrt(
            np.mean(
                [(x - y) ** 2 for x, y in zip(cases_data_fit_past, total_detected_past)]
            )
        ) * stats.norm.ppf(0.5 + q / 2)
        residual_deaths_lb = np.sqrt(
            np.mean(
                [
                    (x - y) ** 2
                    for x, y in zip(deaths_data_fit_past, total_detected_deaths_past)
                ]
            )
        ) * stats.norm.ppf(0.5 - q / 2)
        residual_deaths_ub = np.sqrt(
            np.mean(
                [
                    (x - y) ** 2
                    for x, y in zip(deaths_data_fit_past, total_detected_deaths_past)
                ]
            )
        ) * stats.norm.ppf(0.5 + q / 2)

        # Generation of the dataframe since today
        df_predictions_since_today_cont_country_prov = pd.DataFrame(
            {
                "Continent": [continent for _ in range(n_days_since_today)],
                "Country": [country for _ in range(n_days_since_today)],
                "Province": [province for _ in range(n_days_since_today)],
                "Day": all_dates_since_today,
                "Total Detected": total_detected[n_days_btw_today_since_100:],
                "Active": active_cases[n_days_btw_today_since_100:],
                "Active Hospitalized": active_hospitalized[
                                        n_days_btw_today_since_100:
                                        ],
                "Cumulative Hospitalized": cumulative_hospitalized[
                                            n_days_btw_today_since_100:
                                            ],
                "Total Detected Deaths": total_detected_deaths[
                                            n_days_btw_today_since_100:
                                            ],
                "Active Ventilated": active_ventilated[n_days_btw_today_since_100:],
                "Total Detected True": [np.nan for _ in range(n_days_since_today)],
                "Total Detected Deaths True": [
                    np.nan for _ in range(n_days_since_today)
                ],
                "Total Detected LB": make_increasing([
                    max(int(round(v + residual_cases_lb * np.sqrt(c), 0)), 0)
                    for c, v in enumerate(
                        total_detected[n_days_btw_today_since_100:]
                    )
                ]),
                "Total Detected Deaths LB": make_increasing([
                    max(int(round(v + residual_deaths_lb * np.sqrt(c), 0)), 0)
                    for c, v in enumerate(
                        total_detected_deaths[n_days_btw_today_since_100:]
                    )
                ]),
                "Total Detected UB": [
                    max(int(round(v + residual_cases_ub * np.sqrt(c), 0)), 0)
                    for c, v in enumerate(
                        total_detected[n_days_btw_today_since_100:]
                    )
                ],
                "Total Detected Deaths UB": [
                    max(int(round(v + residual_deaths_ub * np.sqrt(c), 0)), 0)
                    for c, v in enumerate(
                        total_detected_deaths[n_days_btw_today_since_100:]
                    )
                ],
            }
        )
        # Generation of the dataframe from the day since 100th case
        all_dates_since_100 = [
            str((date_day_since100 + timedelta(days=i)).date())
            for i in range(x_sol_final.shape[1])
        ]
        df_predictions_since_100_cont_country_prov = pd.DataFrame(
            {
                "Continent": [
                    continent for _ in range(len(all_dates_since_100))
                ],
                "Country": [country for _ in range(len(all_dates_since_100))],
                "Province": [
                    province for _ in range(len(all_dates_since_100))
                ],
                "Day": all_dates_since_100,
                "Total Detected": total_detected,
                "Active": active_cases,
                "Active Hospitalized": active_hospitalized,
                "Cumulative Hospitalized": cumulative_hospitalized,
                "Total Detected Deaths": total_detected_deaths,
                "Active Ventilated": active_ventilated,
                "Total Detected True": cases_data_fit
                                        + [
                                            np.nan
                                            for _ in range(len(all_dates_since_100) - len(cases_data_fit))
                                        ],
                "Total Detected Deaths True": deaths_data_fit
                                                + [
                                                    np.nan for _ in range(len(all_dates_since_100) - len(deaths_data_fit))
                                                ],
                "Total Detected LB": make_increasing([
                    max(
                        int(round(
                            v + residual_cases_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)),
                            0)
                        ), 0
                    )
                    for c, v in enumerate(total_detected)
                ]),
                "Total Detected Deaths LB": make_increasing([
                    max(
                        int(round(
                            v + residual_deaths_lb * np.sqrt(max(c - n_days_btw_today_since_100, 0)),
                                0)
                        ), 0
                    )
                    for c, v in enumerate(total_detected_deaths)
                ]),
                "Total Detected UB": [
                    max(
                        int(round(
                            v + residual_cases_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)),
                                0)
                        ), 0
                    )
                    for c, v in enumerate(total_detected)
                ],
                "Total Detected Deaths UB": [
                    max(
                        int(round(
                            v + residual_deaths_ub * np.sqrt(max(c - n_days_btw_today_since_100, 0)),
                                0)
                        ), 0
                    )
                    for c, v in enumerate(total_detected_deaths)
                ],
            }
        )
    else:
        df_predictions_since_today_cont_country_prov = pd.DataFrame(
            {
                "Continent": [continent for _ in range(n_days_since_today)],
                "Country": [country for _ in range(n_days_since_today)],
                "Province": [province for _ in range(n_days_since_today)],
                "Day": all_dates_since_today,
                "Total Detected": total_detected[n_days_btw_today_since_100:],
                "Active": active_cases[n_days_btw_today_since_100:],
                "Active Hospitalized": active_hospitalized[
                                        n_days_btw_today_since_100:
                                        ],
                "Cumulative Hospitalized": cumulative_hospitalized[
                                            n_days_btw_today_since_100:
                                            ],
                "Total Detected Deaths": total_detected_deaths[
                                            n_days_btw_today_since_100:
                                            ],
                "Active Ventilated": active_ventilated[n_days_btw_today_since_100:],
                "Total Detected True": [np.nan for _ in range(n_days_since_today)],
                "Total Detected Deaths True": [
                    np.nan for _ in range(n_days_since_today)
                ],
                "Total Detected LB": [np.nan for _ in range(n_days_since_today)],
                "Total Detected Deaths LB": [
                    np.nan for _ in range(n_days_since_today)
                ],
                "Total Detected UB": [np.nan for _ in range(n_days_since_today)],
                "Total Detected Deaths UB": [
                    np.nan for _ in range(n_days_since_today)
                ]
            }
        )
        # Generation of the dataframe from the day since 100th case
        all_dates_since_100 = [
            str((date_day_since100 + timedelta(days=i)).date())
            for i in range(x_sol_final.shape[1])
        ]
        df_predictions_since_100_cont_country_prov = pd.DataFrame(
            {
                "Continent": [
                    continent for _ in range(len(all_dates_since_100))
                ],
                "Country": [country for _ in range(len(all_dates_since_100))],
                "Province": [
                    province for _ in range(len(all_dates_since_100))
                ],
                "Day": all_dates_since_100,
                "Total Detected": total_detected,
                "Active": active_cases,
                "Active Hospitalized": active_hospitalized,
                "Cumulative Hospitalized": cumulative_hospitalized,
                "Total Detected Deaths": total_detected_deaths,
                "Active Ventilated": active_ventilated,
                "Total Detected True": cases_data_fit
                                        + [
                                            np.nan
                                            for _ in range(len(all_dates_since_100) - len(cases_data_fit))
                                        ],
                "Total Detected Deaths True": deaths_data_fit
                                                + [
                                                    np.nan for _ in range(len(all_dates_since_100) - len(deaths_data_fit))
                                                ],
                "Total Detected LB": [
                    np.nan for _ in range(len(all_dates_since_100))
                ],
                "Total Detected Deaths LB": [
                    np.nan for _ in range(len(all_dates_since_100))
                ],
                "Total Detected UB": [
                    np.nan for _ in range(len(all_dates_since_100))
                ],
                "Total Detected Deaths UB": [
                    np.nan for _ in range(len(all_dates_since_100))
                ],
            }
        )
    return (
        df_predictions_since_today_cont_country_prov,
        df_predictions_since_100_cont_country_prov,
    )
