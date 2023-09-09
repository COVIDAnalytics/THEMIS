import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateparser import parse
from copy import deepcopy

from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import read_oxford_country_policy_data, read_policy_data_us_only, get_dominant_policy, get_region_gammas
from pandemic_functions.pandemic_params import future_policies, region_symbol_country_dict, default_dict_normalized_policy_gamma
from cost_functions.economic_cost.economic_data.economic_params import TOTAL_GDP

from sklearn.linear_model import LinearRegression

def get_dominant_policy_df(start_date:str, end_date:str, region:str):
    """Returns dominant policy for every month between start and end date"""
    assert region in region_symbol_country_dict.keys(), "Unidentified region; check pandemic_params"

    country, province = region_symbol_country_dict[region]
    if country == 'US':
        policy_data = read_policy_data_us_only(province, start_date=start_date, end_date=end_date)
    else:
        policy_data = read_oxford_country_policy_data(country=country, start_date=start_date, end_date=end_date)
    
    end_dt = parse(end_date)

    dt1 = parse(start_date)
    dt2 = dt1 + relativedelta(months=1, days=-1)

    dominant_policy_dict = {"month_of": [], "dominant_policy": []}

    while dt2 <= end_dt:
        dominant_policy_dict["month_of"].append(dt1.date())
        dominant_policy_dict["dominant_policy"].append(get_dominant_policy(policy_data, start_date=dt1, end_date=dt2))
        dt1 = dt2 + relativedelta(days=1)
        dt2 = dt1 + relativedelta(months=1, days=-1)

    dominant_policy_df = pd.DataFrame.from_dict(dominant_policy_dict)

    return dominant_policy_df

def get_policy_gdp_impact(region:str, dominant_policy_df:pd.DataFrame):
    """Returns dataframe for the percentage GDP impact for every policy in the region"""
    # Load the data for the monthly GDP impact on region during the pandemic 
    gdp_impact = pd.read_csv(f"cost_functions/economic_cost/economic_data/gdp/{region}.csv")
    gdp_impact["date"]=[datetime(int(r.year), int(r.month), 1).date() for _, r in gdp_impact.iterrows()]
    # Merge with the dominant policy DF
    df = pd.merge(gdp_impact, deepcopy(dominant_policy_df), how="left", left_on="date", right_on="month_of")
    df["GDP"] = df.c + df.i - df.g + df.x
    df.dropna(subset=['dominant_policy'], inplace=True)
    # Load the default policy gamma values to a DF
    policy_gamma_df = pd.DataFrame.from_dict(default_dict_normalized_policy_gamma, orient='index')
    policy_gamma_df.columns = ['gamma']
    # Calculate GDP impact for observed policies
    mean_GDP_impact = 100*(df.groupby("dominant_policy").agg({'c':'mean', 'i':'mean', 'g':'mean', 'x':'mean', 'GDP': 'mean'})/(TOTAL_GDP[region] / (12*1e9) ))
    mean_GDP_impact = mean_GDP_impact.join(policy_gamma_df, how='left')

    # Linear regression to extrapolate for unobserved policies
    x = np.array(mean_GDP_impact.gamma.to_list())
    x = 1-x # reversing gamma to make the line pass through 0 for No Measure
    x = x.reshape(-1,1)
    y = np.array(mean_GDP_impact.GDP.to_list())
    policy_gdp_impact_df = deepcopy(policy_gamma_df)
    if len(y) >= 2:
        model = LinearRegression(fit_intercept=False).fit(x, y)
        r2 = model.score(x,y) # in-sample R^2
        m = model.coef_[0]
        policy_gdp_impact_df["pred_gdp_impact"] = [m*(1-g) for g in policy_gdp_impact_df["gamma"]]
        policy_gdp_impact_df["gdp_r2"] = r2
    else:
        g0 = mean_GDP_impact.iloc[0]['gamma']
        gdp0 = mean_GDP_impact.iloc[0]['GDP']
        policy_gdp_impact_df["pred_gdp_impact"] = [gdp0*(g/g0) for g in policy_gdp_impact_df["gamma"]]
        policy_gdp_impact_df["gdp_r2"] = np.nan

    return policy_gdp_impact_df

def get_policy_employment_impact(region:str, dominant_policy_df:pd.DataFrame):
    """Returns dataframe for the percentage employment impact for every policy in the region"""
    # Load the data for the monthly employment impact on the region during the pandemic
    emp_impact = pd.read_csv(f"cost_functions/economic_cost/economic_data/unemployment/{region}.csv")
    emp_impact["date"]=[datetime(int(r.year), int(r.month), 1).date() for _, r in emp_impact.iterrows()]
    # Merge with dominant policy df
    df = pd.merge(emp_impact, deepcopy(dominant_policy_df), how="left", left_on="date", right_on="month_of")
    # Load the default policy gamma values to a DF
    policy_gamma_df = pd.DataFrame.from_dict(default_dict_normalized_policy_gamma, orient='index')
    policy_gamma_df.columns = ['gamma']
    # Calculate mean employment impact for observed policies
    mean_emp_impact = df.groupby("dominant_policy").agg({'unemployment_gain':'mean'})
    mean_emp_impact = mean_emp_impact.join(policy_gamma_df, how='left')

    # Linear regression to extrapolate to unobserved policies
    x = np.array(mean_emp_impact.gamma.to_list())
    x = 1-x # reversing gamma to make the line pass through 0 for No Measure
    x = x.reshape(-1,1)
    y = np.array(mean_emp_impact.unemployment_gain.to_list())
    policy_employment_impact_df = deepcopy(policy_gamma_df)
    if len(y) >= 2:
        model_emp = LinearRegression(fit_intercept=False).fit(x, y)
        r2 = model_emp.score(x, y) # in-sample R^2
        m_emp = model_emp.coef_[0]
        policy_employment_impact_df["pred_unemployment_gain"] = [m_emp*(1-g) for g in policy_employment_impact_df["gamma"]]
        policy_employment_impact_df["unemployment_r2"] = r2
    else:
        g0 = mean_emp_impact.iloc[0]['gamma']
        u0 = mean_emp_impact.iloc[0]['unemployment_gain']
        policy_employment_impact_df["pred_unemployment_gain"] = [u0*(g/g0) for g in policy_employment_impact_df["gamma"]]
        policy_employment_impact_df["unemployment_r2"] = np.nan

    return policy_employment_impact_df

def get_region_gamma_df(region:str, start_date: str, end_date: str):
    """Returns a DataFrame with the regional gamma values imputed using linear interpolation"""
    region_gamma_dict, reg_results = get_region_gammas(region, start_date=start_date, end_date=end_date, return_regression_result=True)
    df = pd.DataFrame.from_dict(region_gamma_dict, orient='index')
    df.columns = ['region_gamma']
    df['regression_r2'] = reg_results[2]**2
    return df

