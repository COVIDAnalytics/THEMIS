import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timedelta
from policy_functions.policy import Policy
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import run_delphi_policy_scenario
from pandemic_functions.pandemic_params import region_symbol_country_dict, p_v


class Pandemic_Factory:
    def __init__(self):
        self.d_read_data_total_cases = {}
        path_to_predictions_combined = "pandemic_functions/pandemic_data/Global_DELPHI_predictions_combined.csv"
        if os.path.exists(path_to_predictions_combined):
            self.delphi_prediction = pd.read_csv(path_to_predictions_combined)
        else:
            raise FileNotFoundError(f"Can not find file - "+ path_to_predictions_combined + " for actual polcy outcome")
        

    def compute_delphi(self, policy, region):
        country = region_symbol_country_dict[region]
        country_sub = country.replace(' ', '_')
        if country_sub in self.d_read_data_total_cases:
            totalcases = self.d_read_data_total_cases[country_sub]
        else:
            if os.path.exists(f"pandemic_functions/pandemic_data/Cases_{country_sub}_None.csv"):
                totalcases = pd.read_csv(f"pandemic_functions/pandemic_data/Cases_{country_sub}_None.csv")
                self.d_read_data_total_cases[country_sub] = totalcases
            else:
                raise FileNotFoundError(f"Can not find file - pandemic_data/Cases_{country_sub}_None.csv for actual polcy outcome")
        return Pandemic(policy, region, self.delphi_prediction, totalcases)


class Pandemic:
    
    def __init__(self, policy, region, delphi_prediction, totalcases):
    # This is the simulation of the pandemic under a certain policy
    # Given a fixed policy, we can calculate the number of deaths and hospitalizations incurred in such period using DELPHI. 
    # We call it here so that we dont have to repeatedly call DELPHI over and over again. 
        self.policy = policy
        self.region = region
        self.num_cases, self.num_deaths, self.hospitalization_days, self.icu_days, self.ventilated_days = self._get_deaths_and_hospitalizations(delphi_prediction, totalcases)   
        
        
    def _get_deaths_and_hospitalizations(self, delphi_prediction, totalcases):
        # this function gets the number of deaths and hospitalizations that would occur under such policy, using DELPHI
        # the return value is a tuple of numbers

        if self.policy.policy_type == "actual":
            totalcases.date = pd.to_datetime(totalcases.date)
            cases_in_interval = totalcases.query("date >= @start_date and date <= @end_date")
            delphi_prediction.Day = pd.to_datetime(delphi_prediction.Day)
            preds_in_interval = delphi_prediction.query("Day >= @start_date and Day <= @end_date and Country == @country")

            num_deaths = cases_in_interval.iloc[-1]["death_cnt"] - cases_in_interval.iloc[0]["death_cnt"]
            num_cases = cases_in_interval.iloc[-1]["case_cnt"] - cases_in_interval.iloc[0]["case_cnt"]
            hospitalization_days = preds_in_interval["Active Hospitalized"].sum()
            ventilated_days = preds_in_interval["Active Ventilated"].sum()

            if self.region == "GM":
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
            # TODO: implement hypothetical policy case
            num_cases, num_deaths, hospitalization_days, ventilated_days = run_delphi_policy_scenario(self.policy, region_symbol_country_dict[self.region])
            # ventilated_days = hospitalization_days*p_v 
            icu_days = ventilated_days*(0.15/0.85)
            hospitalization_days = hospitalization_days - icu_days
        
        return num_cases, num_deaths, hospitalization_days, icu_days, ventilated_days
        
        


