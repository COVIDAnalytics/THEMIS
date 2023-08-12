# Authors: Hamza Tazi Bouardi (htazi@mit.edu), Michael L. Li (mlli@mit.edu), Omar Skali Lami (oskali@mit.edu), Saksham Soni (sakshams@mit.edu)
import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import compress
from typing import Union, Type
from scipy.integrate import solve_ivp
from scipy import stats
from datetime import datetime, timedelta
# from dateparser import parse
from dateutil.relativedelta import relativedelta

from pandemic_functions.pandemic_params import *
from pandemic_functions.delphi_functions.DELPHI_utils import *
from pandemic_functions.delphi_functions.DELPHI_model import model_covid

import yaml
import os
import argparse

past_parameters = pd.read_csv("pandemic_functions/pandemic_data/Parameters_Global_V2_20200703_with_NY_correction.csv")

def read_oxford_country_policy_data(start_date: str, end_date: str, country: str) -> pd.DataFrame:
    """
    Reads the policy data from the Oxford dataset online and processes it to obtain the MECE policies for a particular
    country (other than the US) within a date range
    :param start_date: string date from which the policy data is collected
    :param end_date: string date till which the policy data is collected
    :return: processed dataframe with MECE policies in each country of the world, used for policy predictions
    """
    measures = raw_measures.copy()
    filtr = ["CountryName", "CountryCode", "Date"]
    target = ["ConfirmedCases", "ConfirmedDeaths"]
    msr = [
        "C1M_School closing",
        "C2M_Workplace closing",
        "C3M_Cancel public events",
        "C4M_Restrictions on gatherings",
        "C5M_Close public transport",
        "C6M_Stay at home requirements",
        "C7M_Restrictions on internal movement",
        "C8EV_International travel controls",
        "H1_Public information campaigns",
    ]

    flags = ["C" + str(i) + "M_Flag" for i in range(1, 8)] + ["H1_Flag"]
    measures = measures.loc[:, filtr + msr + flags + target]

    country_rename = {
            "US": "United States",
            "Korea, South": "South Korea",
            "Congo (Kinshasa)": "Democratic Republic of Congo",
            "Czechia": "Czech Republic",
            "Slovakia": "Slovak Republic",
    }
    if country in country_rename.keys():
        country = country_rename[country]

    measures = measures[measures.CountryName == country]

    measures["Date"] = measures["Date"].apply(
        lambda x: datetime.strptime(str(x), "%Y%m%d")
    )
    for col in target:
        # measures[col] = measures[col].fillna(0)
        measures[col] = measures.groupby("CountryName")[col].ffill()

    measures["C1M_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(measures["C1M_School closing"], measures["C1M_Flag"])
    ]
    measures["C2M_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(measures["C2M_Workplace closing"], measures["C2M_Flag"])
    ]
    measures["C3M_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(measures["C3M_Cancel public events"], measures["C3M_Flag"])
    ]
    measures["C4M_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(
            measures["C4M_Restrictions on gatherings"], measures["C4M_Flag"]
        )
    ]
    measures["C5M_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(measures["C5M_Close public transport"], measures["C5M_Flag"])
    ]
    measures["C6M_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(measures["C6M_Stay at home requirements"], measures["C6M_Flag"])
    ]
    measures["C7M_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(
            measures["C7M_Restrictions on internal movement"], measures["C7M_Flag"]
        )
    ]
    measures["H1_Flag"] = [
        0 if x <= 0 else y
        for (x, y) in zip(
            measures["H1_Public information campaigns"], measures["H1_Flag"]
        )
    ]

    measures["C1M_School closing"] = [
        int(a and b)
        for a, b in zip(measures["C1M_School closing"] >= 2, measures["C1M_Flag"] == 1)
    ]

    measures["C2M_Workplace closing"] = [
        int(a and b)
        for a, b in zip(measures["C2M_Workplace closing"] >= 2, measures["C2M_Flag"] == 1)
    ]

    measures["C3M_Cancel public events"] = [
        int(a and b)
        for a, b in zip(
            measures["C3M_Cancel public events"] >= 2, measures["C3M_Flag"] == 1
        )
    ]

    measures["C4M_Restrictions on gatherings"] = [
        int(a and b)
        for a, b in zip(
            measures["C4M_Restrictions on gatherings"] >= 1, measures["C4M_Flag"] == 1
        )
    ]

    measures["C5M_Close public transport"] = [
        int(a and b)
        for a, b in zip(
            measures["C5M_Close public transport"] >= 2, measures["C5M_Flag"] == 1
        )
    ]

    measures["C6M_Stay at home requirements"] = [
        int(a and b)
        for a, b in zip(
            measures["C6M_Stay at home requirements"] >= 2, measures["C6M_Flag"] == 1
        )
    ]

    measures["C7M_Restrictions on internal movement"] = [
        int(a and b)
        for a, b in zip(
            measures["C7M_Restrictions on internal movement"] >= 2,
            measures["C7M_Flag"] == 1,
        )
    ]

    measures["C8EV_International travel controls"] = [
        int(a) for a in (measures["C8EV_International travel controls"] >= 3)
    ]

    measures["H1_Public information campaigns"] = [
        int(a and b)
        for a, b in zip(
            measures["H1_Public information campaigns"] >= 1, measures["H1_Flag"] == 1
        )
    ]

    # measures = measures.loc[:, measures.isnull().mean() < 0.1]
    msr = set(measures.columns).intersection(set(msr))

    # measures = measures.fillna(0)
    measures = measures.dropna()
    for col in msr:
        measures[col] = measures[col].apply(lambda x: int(x > 0))
    measures = measures[["CountryName", "Date"] + list(sorted(msr))]

    measures = measures.fillna(0)
    msr = future_policies

    measures["Restrict_Mass_Gatherings"] = [
        int(a or b or c)
        for a, b, c in zip(
            measures["C3M_Cancel public events"],
            measures["C4M_Restrictions on gatherings"],
            measures["C5M_Close public transport"],
        )
    ]
    measures["Others"] = [
        int(a or b or c)
        for a, b, c in zip(
            measures["C2M_Workplace closing"],
            measures["C7M_Restrictions on internal movement"],
            measures["C8EV_International travel controls"],
        )
    ]

    del measures["C2M_Workplace closing"]
    del measures["C3M_Cancel public events"]
    del measures["C4M_Restrictions on gatherings"]
    del measures["C5M_Close public transport"]
    del measures["C7M_Restrictions on internal movement"]
    del measures["C8EV_International travel controls"]

    output = deepcopy(measures)
    output[msr[0]] = (measures.iloc[:, 2:].sum(axis=1) == 0).apply(lambda x: int(x))
    output[msr[1]] = [
        int(a and b)
        for a, b in zip(
            measures.iloc[:, 2:].sum(axis=1) == 1,
            measures["Restrict_Mass_Gatherings"] == 1,
        )
    ]
    output[msr[2]] = [
        int(a and b and c)
        for a, b, c in zip(
            measures.iloc[:, 2:].sum(axis=1) > 0,
            measures["Restrict_Mass_Gatherings"] == 0,
            measures["C6M_Stay at home requirements"] == 0,
        )
    ]
    output[msr[3]] = [
        int(a and b and c)
        for a, b, c in zip(
            measures.iloc[:, 2:].sum(axis=1) == 2,
            measures["C1M_School closing"] == 1,
            measures["Restrict_Mass_Gatherings"] == 1,
        )
    ]
    output[msr[4]] = [
        int(a and b and c and d)
        for a, b, c, d in zip(
            measures.iloc[:, 2:].sum(axis=1) > 1,
            measures["C1M_School closing"] == 0,
            measures["Restrict_Mass_Gatherings"] == 1,
            measures["C6M_Stay at home requirements"] == 0,
        )
    ]
    output[msr[5]] = [
        int(a and b and c and d)
        for a, b, c, d in zip(
            measures.iloc[:, 2:].sum(axis=1) > 2,
            measures["C1M_School closing"] == 1,
            measures["Restrict_Mass_Gatherings"] == 1,
            measures["C6M_Stay at home requirements"] == 0,
        )
    ]
    output[msr[6]] = (measures["C6M_Stay at home requirements"] == 1).apply(
        lambda x: int(x)
    )
    output.rename(columns={"CountryName": "country", "Date": "date"}, inplace=True)
    output["province"] = "None"
    output = output.loc[:, ["country", "province", "date"] + msr]
    output = output[(output.date >= start_date) & (output.date <= end_date)].reset_index(drop=True)
    return output

def convert_dates_us_policies(raw_date: str) -> Union[float, datetime]:
    """
    Converts dates from the dataframe with raw policies implemented in the US
    :param raw_date: a certain date string in a raw format
    :return: a datetime in the right format for the final policy dataframe
    """
    if raw_date == "Not implemented":
        return np.nan
    else:
        x_long = raw_date + "20"
        return pd.to_datetime(x_long, format="%d-%b-%Y")

def check_us_policy_data_consistency(policies: list, df_policy_raw_us: pd.DataFrame):
    """
    Checks consistency of the policy data in the US retrieved e.g. from IHME by verifying that if there is an end date
    there must also be a start date for the policy implemented
    :param policies: list of policies under consideration
    :param df_policy_raw_us: slightly processed dataframe with policies implemented in the US
    :return:
    """
    for policy in policies:
        assert (
            len(
                df_policy_raw_us.loc[
                    (df_policy_raw_us[f"{policy}_start_date"].isnull())
                    & (~df_policy_raw_us[f"{policy}_end_date"].isnull()),
                    :,
                ]
            )
            == 0
        ), f"Problem in data, policy {policy} has no start date but has an end date"

def create_intermediary_policy_features_us(
    df_policy_raw_us: pd.DataFrame, dict_state_to_policy_dates: dict, policies: list,
    policies_end_date: str = '20220101'
) -> pd.DataFrame:
    """
    Processes the IHME policy data in the US to create the right intermediary features with the right names
    :param df_policy_raw_us: raw dataframe with policies implemented in the US
    :param dict_state_to_policy_dates: dictionary of the format {state: {policy: [start_date, end_date]}}
    :param policies: list of policies under consideration
    :return: an intermediary dataframe with processed columns containing binary variables as to whether or not a
    policy is implemented in a given state at a given date
    """
    list_df_concat = []
    max_date = pd.to_datetime(policies_end_date)
    n_dates = (max_date - datetime(2020, 3, 1)).days + 1
    date_range = [datetime(2020, 3, 1) + timedelta(days=i) for i in range(n_dates)]
    for location in df_policy_raw_us.location_name.unique():
        df_temp = pd.DataFrame(
            {
                "continent": ["North America" for _ in range(len(date_range))],
                "country": ["US" for _ in range(len(date_range))],
                "province": [location for _ in range(len(date_range))],
                "date": date_range,
            }
        )
        for policy in policies:
            start_date_policy_location = dict_state_to_policy_dates[location][policy][0]
            start_date_policy_location = (
                start_date_policy_location
                if start_date_policy_location is not np.nan
                else "2030-01-02"
            )
            end_date_policy_location = dict_state_to_policy_dates[location][policy][1]
            end_date_policy_location = (
                end_date_policy_location
                if end_date_policy_location is not np.nan
                else "2030-01-01"
            )
            df_temp[policy] = 0
            df_temp.loc[
                (
                    (df_temp.date >= start_date_policy_location)
                    & (df_temp.date <= end_date_policy_location)
                ),
                policy,
            ] = 1

        list_df_concat.append(df_temp)

    df_policies_US = pd.concat(list_df_concat).reset_index(drop=True)
    df_policies_US.rename(
        columns={
            "travel_limit": "Travel_severely_limited",
            "stay_home": "Stay_at_home_order",
            "educational_fac": "Educational_Facilities_Closed",
            "any_gathering_restrict": "Mass_Gathering_Restrictions",
            "any_business": "Initial_Business_Closure",
            "all_non-ess_business": "Non_Essential_Services_Closed",
        },
        inplace=True,
    )
    return df_policies_US

def create_final_policy_features_us(df_policies_US: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the final MECE policies in the US from the intermediary policies dataframe
    :param df_policies_US: intermediary dataframe with processed columns containing binary variables as to whether or 
    not a policy is implemented in a given state at a given date
    :return: dataframe with the final MECE policies in the US used for DELPHI policy predictions
    """
    df_policies_US_final = deepcopy(df_policies_US)
    msr = future_policies
    df_policies_US_final[msr[0]] = (df_policies_US.sum(axis=1) == 0).apply(
        lambda x: int(x)
    )
    df_policies_US_final[msr[1]] = [
        int(a and b)
        for a, b in zip(
            df_policies_US.sum(axis=1) == 1,
            df_policies_US["Mass_Gathering_Restrictions"] == 1,
        )
    ]
    df_policies_US_final[msr[2]] = [
        int(a and b and c)
        for a, b, c in zip(
            df_policies_US.sum(axis=1) > 0,
            df_policies_US["Mass_Gathering_Restrictions"] == 0,
            df_policies_US["Stay_at_home_order"] == 0,
        )
    ]
    df_policies_US_final[msr[3]] = [
        int(a and b and c)
        for a, b, c in zip(
            df_policies_US.sum(axis=1) == 2,
            df_policies_US["Educational_Facilities_Closed"] == 1,
            df_policies_US["Mass_Gathering_Restrictions"] == 1,
        )
    ]
    df_policies_US_final[msr[4]] = [
        int(a and b and c and d)
        for a, b, c, d in zip(
            df_policies_US.sum(axis=1) > 1,
            df_policies_US["Educational_Facilities_Closed"] == 0,
            df_policies_US["Mass_Gathering_Restrictions"] == 1,
            df_policies_US["Stay_at_home_order"] == 0,
        )
    ]
    df_policies_US_final[msr[5]] = [
        int(a and b and c and d)
        for a, b, c, d in zip(
            df_policies_US.sum(axis=1) > 2,
            df_policies_US["Educational_Facilities_Closed"] == 1,
            df_policies_US["Mass_Gathering_Restrictions"] == 1,
            df_policies_US["Stay_at_home_order"] == 0,
        )
    ]
    df_policies_US_final[msr[6]] = (df_policies_US["Stay_at_home_order"] == 1).apply(
        lambda x: int(x)
    )
    df_policies_US_final["country"] = "US"
    df_policies_US_final = df_policies_US_final.loc[
        :, ["country", "province", "date"] + msr
    ]
    return df_policies_US_final

def read_policy_data_us_only(state: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Reads and processes the policy data from IHME to obtain the MECE policies defined for DELPHI Policy Predictions
    :param filepath_pandemic_data: string, path to the pandemic data folder
    :param state: string, state for which policy scenarios are being run
    :return: fully processed dataframe containing the MECE policies implemented in each state of the US for the full 
    time period necessary until the day when this function is called
    """
    policies = [
        "travel_limit", "stay_home", "educational_fac", "any_gathering_restrict",
        "any_business", "all_non-ess_business",
    ]
    df = df_raw_us_policies[df_raw_us_policies.location_name == state][
        [
            "location_name", "travel_limit_start_date", "travel_limit_end_date", "stay_home_start_date",
            "stay_home_end_date", "educational_fac_start_date", "educational_fac_end_date",
            "any_gathering_restrict_start_date", "any_gathering_restrict_end_date", "any_business_start_date",
            "any_business_end_date", "all_non-ess_business_start_date", "all_non-ess_business_end_date",
        ]
    ]
    dict_state_to_policy_dates = {}
    for location in df.location_name.unique():
        df_temp = df[df.location_name == location].reset_index(drop=True)
        dict_state_to_policy_dates[location] = {
            policy: [
                df_temp.loc[0, f"{policy}_start_date"],
                df_temp.loc[0, f"{policy}_end_date"],
            ]
            for policy in policies
        }
    check_us_policy_data_consistency(policies=policies, df_policy_raw_us=df)
    df_policies_US = create_intermediary_policy_features_us(
        df_policy_raw_us=df,
        dict_state_to_policy_dates=dict_state_to_policy_dates,
        policies=policies,
    )
    df_policies_US_final = create_final_policy_features_us(
        df_policies_US=df_policies_US
    )
    df_policies_US_final = df_policies_US_final[(df_policies_US_final.date >= start_date) & (df_policies_US_final.date <= end_date)]

    return df_policies_US_final

def get_latest_policy(policy_data: pd.DataFrame, start_date: datetime) -> list:
    latest_policy = list(
        compress(
            future_policies, 
            (
                policy_data[
                    policy_data.date == start_date
                ][future_policies] == 1
            ).values
            .flatten()
            .tolist()
        )
    )[0]

    return latest_policy

def get_dominant_policy(policy_data: pd.DataFrame, start_date: datetime, end_date: datetime):
    """Provides a Dominant Policy between start_date and end_date, both inclusive"""
    policy_data_range = policy_data[(policy_data.date >= start_date) & (policy_data.date <= end_date)]
    policies = []
    for i, policy in enumerate(future_policies):
        count = policy_data_range[policy].sum()
        policies.extend( [i]*count )
    
    return future_policies[ int(np.median(policies)) ]

def get_region_gammas(region: str, policy_days_thresh: int = 10) -> dict:
    """
    Function to calculate the gamma values for the region by interpolating the values for the observed policies using the default gamma values
    Parameters:
        - pandemic: Pandemic object containing the information of the region and duration that is being analyzed
    Returns:
        - Dict of policy -> gamma avalue
    """
    country, province = region_symbol_country_dict[region]
    params_list = past_parameters.query("Country == @country and Province == @province")[
        ["Data Start Date", "Median Day of Action", "Rate of Action", "Jump Magnitude", "Jump Time", "Jump Decay"]
    ].iloc[0]
    # final_start_date = max(parse(params_list['Data Start Date']), parse(policy_data_start_date))
    # final_start_date = str(final_start_date.date())
    final_start_date=policy_data_start_date

    if country == 'US':
        policy_data = read_policy_data_us_only(state=province, start_date=final_start_date, end_date=policy_data_end_date)
    else:
        policy_data = read_oxford_country_policy_data(country=country, start_date=final_start_date, end_date=policy_data_end_date)

    policy_data.loc[:, "Gamma"] = [
        gamma_t(day, params_list)
        for day in policy_data["date"]
    ]
    n_measures = policy_data.iloc[:, 3:-1].shape[1]
    dict_region_policy_gamma = {
        policy_data.columns[3 + i]: policy_data[
            policy_data.iloc[:, 3 + i] == 1
        ]
        .iloc[:, -1]
        .mean()
        for i in range(n_measures)
    }
    dict_region_policy_counts = {
        policy_data.columns[3 + i]: policy_data[
            policy_data.iloc[:, 3 + i] == 1
        ]
        .iloc[:, 3 + i]
        .sum()
        for i in range(n_measures)
    }
    dict_region_policy_gamma = dict(sorted(dict_region_policy_gamma.items(), key=lambda x: x[0]))
    dict_region_policy_counts = dict(sorted(dict_region_policy_counts.items(), key=lambda x: x[0]))
    default_policy_gammas = deepcopy(default_dict_normalized_policy_gamma)
    default_policy_gammas = dict(sorted(default_policy_gammas.items(), key=lambda x: x[0]))

    from scipy.stats import linregress

    x = np.array(list(default_policy_gammas.values()))
    y = np.array(list(dict_region_policy_gamma.values()))
    ind = np.array(list(dict_region_policy_counts.values()))

    train_keys = np.array(list(default_policy_gammas.keys()))
    train_keys = train_keys[(~np.isnan(y)) & (ind > policy_days_thresh)]

    assert len(train_keys) >= 2, "Not enough data about policies to run simulation"

    x_train = x[(~np.isnan(y)) & (ind > policy_days_thresh)]
    y_train = y[(~np.isnan(y)) & (ind > policy_days_thresh)]
    y_train = sigmoid_inv_np(y_train/2)

    m, C, _, _, _ = linregress(x_train, y_train)

    for key in dict_region_policy_gamma.keys():
        if key not in train_keys:
            dict_region_policy_gamma[key] = 2*sigmoid(m*default_policy_gammas[key] + C)

    return dict_region_policy_gamma

def run_delphi_policy_scenario(policy, region, totalcases, dict_region_policy_gamma):
    country, province = region_symbol_country_dict[region]
    parameter_list_total = past_parameters[(past_parameters.Country == country) & (past_parameters.Province == province)]

    if len(parameter_list_total) > 0:
        parameter_list_line = parameter_list_total.iloc[-1, :].values.tolist()
        parameter_list = parameter_list_line[5:]
        date_day_since100 = pd.to_datetime(parameter_list_line[3])
        # Allowing a 5% drift for states with past predictions, starting in the 5th position are the parameters
        start_date = date_day_since100
        validcases = totalcases[
            (totalcases.date >= str(date_day_since100.date()))
            & (totalcases.date <= str(policy.end_date.date()))
        ][["day_since100", "case_cnt", "death_cnt", "total_hospitalization", "people_vaccinated", "people_fully_vaccinated"]].reset_index(drop=True)
    else:
        print(f"Couldn't find past parameters for {country} and {province}")
        return 0, 0, 0, 0

    # Now we start the modeling part:
    if len(validcases) > validcases_threshold_policy:
        PopulationT = global_populations[
            (global_populations.Country == country) & (global_populations.Province == province)
        ].pop2016.iloc[-1]
        N = PopulationT
        PopulationI = validcases.loc[0, "case_cnt"]
        PopulationD = validcases.loc[0, "death_cnt"]
        PopulationR = validcases.loc[0, "death_cnt"] * 5 if validcases.loc[0, "case_cnt"] - validcases.loc[0, "death_cnt"]> validcases.loc[0, "death_cnt"] * 5 else 0

        policy_scenario_end_date = pd.to_datetime(policy.end_date)
        maxT = (policy_scenario_end_date - date_day_since100).days + 1
        policy_scenario_start_date = pd.to_datetime(policy.start_date)
        if policy_scenario_start_date < date_day_since100:
            raise ValueError("Policy start date too early for DELPHI to model the epidemic")
        policy_startT = (policy_scenario_start_date - date_day_since100).days + 1
        # policy_startT = max((policy_scenario_start_date - date_day_since100).days + 1, 1)
        t_cases = validcases["day_since100"].tolist() - validcases.loc[0, "day_since100"]
        GLOBAL_PARAMS_FIXED = (N, PopulationR, PopulationD, PopulationI, p_v, p_d, p_h)
        best_params = parameter_list
        t_predictions = list(range(maxT))

        policy_scenario_gammas = {}
        for i, alt_policy in enumerate(policy.policy_vector):
            start = policy_scenario_start_date + relativedelta(months=i)
            end = policy_scenario_start_date + relativedelta(months=i+1)
            if start >= date_day_since100:
                t1 = (start - date_day_since100).days + 1
                t2 = (end - date_day_since100).days + 1
                policy_scenario_gammas[(t1, t2)] = dict_region_policy_gamma[alt_policy]

        # Variables Initialization for the ODE system
        x_0_cases = get_initial_conditions(
            params_fitted=best_params,
            global_params_fixed=GLOBAL_PARAMS_FIXED
        )

        # Solve IVP, define model as a function of only the fitted params
        def model_covid_predictions(t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, 
                k1, k2, jump, t_jump, std_normal):
            return model_covid(t, x, alpha, days, r_s, r_dth, p_dth, r_dthdecay, 
                k1, k2, jump, t_jump, std_normal, N, policy_scenario_gammas, policy_startT, maxT)
        x_sol_final = solve_ivp(
            fun=model_covid_predictions,
            y0=x_0_cases,
            t_span=[t_predictions[0], t_predictions[-1]],
            t_eval=t_predictions,
            args=tuple(best_params),
        ).y

        num_cases = round(x_sol_final[15,-1] - x_sol_final[15, policy_startT-1], 0)
        num_deaths = round(x_sol_final[14, -1] - x_sol_final[14, policy_startT-1], 0)
        active_hospitalized = (
                x_sol_final[4, :] + x_sol_final[7, :]
        )  # DHR + DHD
        active_hospitalized = [int(round(x, 0)) for x in active_hospitalized]
        active_ventilated = (
                x_sol_final[12, :] + x_sol_final[13, :]
        )  # DVR + DVD
        active_ventilated = [int(round(x, 0)) for x in active_ventilated]
        hospitalization_days = sum(active_hospitalized[(policy_startT-1):])
        ventilated_days = sum(active_ventilated[(policy_startT-1):])

    ## TODO:
    ## - return LB UB in results
    return num_cases, num_deaths, hospitalization_days, ventilated_days
