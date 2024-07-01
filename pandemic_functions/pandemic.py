import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
from policy_functions.policy import Policy
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import run_delphi_policy_scenario, get_region_gammas, get_region_gammas_v2
from pandemic_functions.pandemic_params import region_symbol_country_dict, p_v


class Pandemic_Factory:
    """A wrapper class to help with the loading of different parameters and computing the Pandemic object"""
    def __init__(self):
        self.d_read_data_total_cases = {}
        self.d_region_policy_gammas = {}
        path_to_predictions_combined = "pandemic_functions/pandemic_data/Global_DELPHI_predictions_combined.csv"
        if os.path.exists(path_to_predictions_combined):
            self.delphi_prediction = pd.read_csv(path_to_predictions_combined)
        else:
            raise FileNotFoundError(f"Can not find file - "+ path_to_predictions_combined + " for actual polcy outcome")
        

    def compute_delphi(self, policy, region, **kwargs):
        """
        Loads the required data and simulates the pandemic using DELHPI for the given policy.
        Returns an object of type `Pandemic`.
        """
        country, province = region_symbol_country_dict[region]
        country_sub = country.replace(' ', '_')
        province_sub = province.replace(' ', '_')
        if country_sub in self.d_read_data_total_cases:
            totalcases = self.d_read_data_total_cases[region]
            dict_region_policy_gamma = self.d_region_policy_gammas[region]
        else:
            if os.path.exists(f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv"):
                totalcases = pd.read_csv(f"pandemic_functions/pandemic_data/Cases_{country_sub}_{province_sub}.csv")
                # dict_region_policy_gamma = get_region_gammas(region)
                dict_region_policy_gamma = get_region_gammas_v2(region)
                self.d_read_data_total_cases[region] = totalcases
                self.d_region_policy_gammas[region] = dict_region_policy_gamma
            else:
                raise FileNotFoundError(f"Can not find file - pandemic_data/Cases_{country_sub}_{province_sub}.csv for actual polcy outcome")
        return Pandemic(policy, region, self.delphi_prediction, totalcases, dict_region_policy_gamma, **kwargs)


class Pandemic:
    """
    Class to encapsulate the pandemic scenario. When initialized with the optional parameter `sample_gammas=True` 
    and if the policy is not actual, it will sample gammas from a distribution to compute lower and upper bounds on predictions.
    The optional parameter `n_sample:int` is used to specify the number of times gamma is sampled (default = 20).
    """
    def __init__(self, policy, region, delphi_prediction, totalcases, dict_region_policy_gamma, **kwargs):
    # This is the simulation of the pandemic under a certain policy
    # Given a fixed policy, we can calculate the number of deaths and hospitalizations incurred in such period using DELPHI. 
    # We call it here so that we dont have to repeatedly call DELPHI over and over again. 
        self.policy = policy
        self.region = region
        output = self._get_deaths_and_hospitalizations(delphi_prediction, totalcases, dict_region_policy_gamma, **kwargs)
        self.num_cases, self.num_cases_lb, self.num_cases_ub, self.num_deaths, self.num_deaths_lb, \
            self.num_deaths_ub, self.hospitalization_days, self.hospitalization_days_lb, self.hospitalization_days_ub, \
            self.icu_days, self.icu_days_lb, self.icu_days_ub, self.ventilated_days, self.ventilated_days_lb, \
            self.ventilated_days_ub = output   
        
        
    def _get_deaths_and_hospitalizations(self, delphi_prediction, totalcases, dict_region_policy_gamma, 
                                        sample_gammas:bool=False, n_sample:int=20):
        # this function gets the number of deaths and hospitalizations that would occur under such policy, using DELPHI
        # the return value is a tuple of numbers
        country, province = region_symbol_country_dict[self.region]
        if self.policy.policy_type == "actual":
            totalcases.date = pd.to_datetime(totalcases.date)
            start_date = pd.to_datetime(self.policy.start_date)
            end_date = start_date + pd.DateOffset(months=self.policy.num_months)
            cases_in_interval = totalcases.query("date >= @start_date and date <= @end_date")
            delphi_prediction.Day = pd.to_datetime(delphi_prediction.Day)
            preds_in_interval = delphi_prediction.query("Day >= @start_date and Day <= @end_date and Country == @country and Province == @province")

            num_deaths = cases_in_interval.iloc[-1]["death_cnt"] - cases_in_interval.iloc[0]["death_cnt"]
            num_cases = cases_in_interval.iloc[-1]["case_cnt"] - cases_in_interval.iloc[0]["case_cnt"]
            hospitalization_days = preds_in_interval["Active Hospitalized"].sum()
            ventilated_days = preds_in_interval["Active Ventilated"].sum()
            num_cases_lb = num_cases_ub = num_deaths_lb = num_deaths_ub = np.nan
            hospitalization_days_lb = hospitalization_days_ub = icu_days_lb = icu_days_ub = ventilated_days_lb = ventilated_days_ub = np.nan

            if self.region == "DE":
                hosp_global = pd.read_csv("pandemic_functions/pandemic_data/global_hospitalizations.csv")
                hosp_global.date = pd.to_datetime(hosp_global.date)
                hosp_germany = hosp_global[hosp_global.country_id == "DE"].copy()
                hosp_germany.date = pd.to_datetime(hosp_germany.date)
                hosp_germany = hosp_germany.query("date >= @start_date and date <= @end_date")
                icu_days = np.nansum(hosp_germany.icu_beds_used)
                icu_days = icu_days - ventilated_days
                hospitalization_days = hospitalization_days - icu_days
            else:
                icu_days = ventilated_days*(0.15/0.85) # (1/0.85 - 1)*ventilated_days
                hospitalization_days = hospitalization_days - icu_days
        else:
            num_cases, num_cases_lb, num_cases_ub, num_deaths, num_deaths_lb, num_deaths_ub, \
                    hospitalization_days, ventilated_days = run_delphi_policy_scenario(self.policy, self.region, totalcases, dict_region_policy_gamma)
            hospitalization_days_lb, ventilated_days_lb = hospitalization_days, ventilated_days
            hospitalization_days_ub, ventilated_days_ub = hospitalization_days, ventilated_days
            if sample_gammas:
                # gamma_samples = get_region_gammas(self.region, sample_gammas=True, n_sample=n_sample)
                gamma_samples = get_region_gammas_v2(self.region, sample_gammas=True, n_sample=n_sample)
                for dict_gammas in gamma_samples:
                    _, nclb, ncub, _, ndlb, ndub, nhd, nvd = run_delphi_policy_scenario(self.policy, self.region, totalcases, dict_gammas)
                    num_cases_lb = min(num_cases_lb, nclb)
                    num_cases_ub = max(num_cases_ub, ncub)
                    num_deaths_lb = min(num_deaths_lb, ndlb)
                    num_deaths_ub = max(num_deaths_ub, ndub)
                    hospitalization_days_lb = min(hospitalization_days_lb, nhd)
                    hospitalization_days_ub = max(hospitalization_days_ub, nhd)
                    ventilated_days_lb = min(ventilated_days_lb, nvd)
                    ventilated_days_ub = max(ventilated_days_ub, nvd)
            # ventilated_days = hospitalization_days*p_v 
            icu_vec = np.array([ventilated_days, ventilated_days_lb, ventilated_days_ub])*(0.15/0.85)
            icu_days, icu_days_lb, icu_days_ub = tuple(icu_vec)
            hospitalization_days, hospitalization_days_lb, hospitalization_days_ub = \
                tuple(np.array([hospitalization_days, hospitalization_days_lb, hospitalization_days_ub]) - icu_vec)
        
        return num_cases, num_cases_lb, num_cases_ub, num_deaths, num_deaths_lb, num_deaths_ub, \
            hospitalization_days, hospitalization_days_lb, hospitalization_days_ub, \
            icu_days, icu_days_lb, icu_days_ub, ventilated_days, ventilated_days_lb, ventilated_days_ub
        
        


